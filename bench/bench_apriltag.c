/*
 * bench_apriltag.c -- Benchmark the apriltag library against a ground truth dataset.
 *
 * Loads grayscale PNG images, runs apriltag detection multiple times per image,
 * measures wall-clock timing, verifies accuracy against ground truth JSON files,
 * and outputs results as CSV + summary JSON.
 */

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#include "stb_image.h"
#include "cJSON.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dirent.h>
#include <getopt.h>
#include <stdbool.h>
#include <errno.h>

#include "apriltag.h"
#include "tag36h11.h"
#include "common/image_u8.h"
#include "common/zarray.h"
#include "common/timeprofile.h"

/* ------------------------------------------------------------------ */
/* Configuration                                                       */
/* ------------------------------------------------------------------ */

typedef struct {
    const char *data_dir;
    int iterations;
    int warmup;
    int nthreads;
    float quad_decimate;
    float quad_sigma;
    int refine_edges;
    double decode_sharpening;
    int hamming_bits;
    const char *output_csv;
    const char *summary_json;
    int quiet;
    int limit;
    int verify;
} bench_config_t;

/* ------------------------------------------------------------------ */
/* Ground truth types                                                  */
/* ------------------------------------------------------------------ */

typedef struct {
    int tag_id;
    int hamming;
    double decision_margin;
    double center[2];
    double corners[4][2];
} gt_detection_t;

typedef struct {
    char filename[512];
    int image_width;
    int image_height;
    int num_detections;
    gt_detection_t *detections;
} gt_image_t;

/* ------------------------------------------------------------------ */
/* Per-image result                                                    */
/* ------------------------------------------------------------------ */

typedef struct {
    char filename[512];
    int image_width;
    int image_height;
    int gt_num_detections;
    int det_num_detections;
    int tags_correct;
    int tags_missed;
    int tags_extra;
    double max_center_err;
    double max_corner_err;
    double min_ms;
    double max_ms;
    double mean_ms;
    double median_ms;
    double stddev_ms;
} bench_result_t;

/* ------------------------------------------------------------------ */
/* Timing                                                              */
/* ------------------------------------------------------------------ */

static inline double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}

/* ------------------------------------------------------------------ */
/* Statistics                                                          */
/* ------------------------------------------------------------------ */

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

typedef struct {
    double min, max, mean, median, stddev;
} timing_stats_t;

static timing_stats_t compute_stats(double *times, int n) {
    timing_stats_t s = {0};
    if (n <= 0) return s;
    qsort(times, n, sizeof(double), cmp_double);
    s.min = times[0];
    s.max = times[n - 1];
    s.median = (n % 2) ? times[n / 2] : (times[n / 2 - 1] + times[n / 2]) / 2.0;
    double sum = 0;
    for (int i = 0; i < n; i++) sum += times[i];
    s.mean = sum / n;
    double var = 0;
    for (int i = 0; i < n; i++) var += (times[i] - s.mean) * (times[i] - s.mean);
    s.stddev = sqrt(var / n);
    return s;
}

static double percentile(double *sorted, int n, double p) {
    if (n <= 0) return 0;
    double idx = p * (n - 1);
    int lo = (int)idx;
    int hi = lo + 1;
    if (hi >= n) return sorted[n - 1];
    double frac = idx - lo;
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

/* ------------------------------------------------------------------ */
/* PNG loading -> image_u8_t                                           */
/* ------------------------------------------------------------------ */

static image_u8_t *load_png_as_image_u8(const char *path) {
    int w, h, channels;
    unsigned char *pixels = stbi_load(path, &w, &h, &channels, 1);
    if (!pixels) return NULL;

    image_u8_t *im = image_u8_create(w, h);
    if (!im) {
        stbi_image_free(pixels);
        return NULL;
    }

    for (int y = 0; y < h; y++) {
        memcpy(&im->buf[y * im->stride], &pixels[y * w], w);
    }

    stbi_image_free(pixels);
    return im;
}

/* ------------------------------------------------------------------ */
/* Ground truth JSON parsing                                           */
/* ------------------------------------------------------------------ */

static int parse_ground_truth(const char *json_path, gt_image_t *gt) {
    FILE *f = fopen(json_path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(len + 1);
    if (!buf) { fclose(f); return -1; }
    size_t nread = fread(buf, 1, len, f);
    buf[nread] = '\0';
    fclose(f);

    cJSON *root = cJSON_Parse(buf);
    free(buf);
    if (!root) return -1;

    cJSON *item;

    item = cJSON_GetObjectItem(root, "image_file");
    if (item && item->valuestring)
        strncpy(gt->filename, item->valuestring, sizeof(gt->filename) - 1);

    item = cJSON_GetObjectItem(root, "image_width");
    gt->image_width = item ? item->valueint : 0;

    item = cJSON_GetObjectItem(root, "image_height");
    gt->image_height = item ? item->valueint : 0;

    item = cJSON_GetObjectItem(root, "num_detections");
    gt->num_detections = item ? item->valueint : 0;

    cJSON *dets = cJSON_GetObjectItem(root, "detections");
    if (!dets || gt->num_detections == 0) {
        gt->detections = NULL;
        gt->num_detections = 0;
        cJSON_Delete(root);
        return 0;
    }

    gt->detections = calloc(gt->num_detections, sizeof(gt_detection_t));
    if (!gt->detections) {
        cJSON_Delete(root);
        return -1;
    }

    for (int i = 0; i < gt->num_detections; i++) {
        cJSON *det = cJSON_GetArrayItem(dets, i);
        gt_detection_t *d = &gt->detections[i];

        item = cJSON_GetObjectItem(det, "tag_id");
        d->tag_id = item ? item->valueint : -1;

        item = cJSON_GetObjectItem(det, "hamming");
        d->hamming = item ? item->valueint : 0;

        item = cJSON_GetObjectItem(det, "decision_margin");
        d->decision_margin = item ? item->valuedouble : 0;

        cJSON *center = cJSON_GetObjectItem(det, "center");
        if (center) {
            d->center[0] = cJSON_GetArrayItem(center, 0)->valuedouble;
            d->center[1] = cJSON_GetArrayItem(center, 1)->valuedouble;
        }

        cJSON *corners = cJSON_GetObjectItem(det, "corners");
        if (corners) {
            for (int c = 0; c < 4; c++) {
                cJSON *corner = cJSON_GetArrayItem(corners, c);
                d->corners[c][0] = cJSON_GetArrayItem(corner, 0)->valuedouble;
                d->corners[c][1] = cJSON_GetArrayItem(corner, 1)->valuedouble;
            }
        }
    }

    cJSON_Delete(root);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Image discovery                                                     */
/* ------------------------------------------------------------------ */

static int cmp_str(const void *a, const void *b) {
    return strcmp(*(const char **)a, *(const char **)b);
}

static int discover_images(const char *data_dir, char ***out_files) {
    char img_dir[1024];
    snprintf(img_dir, sizeof(img_dir), "%s/images", data_dir);

    DIR *dir = opendir(img_dir);
    if (!dir) {
        fprintf(stderr, "ERROR: Cannot open image directory: %s\n", img_dir);
        return 0;
    }

    int capacity = 4096;
    char **files = malloc(capacity * sizeof(char *));
    int count = 0;

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        const char *name = entry->d_name;
        size_t len = strlen(name);
        if (len > 4 && strcmp(name + len - 4, ".png") == 0) {
            if (count >= capacity) {
                capacity *= 2;
                files = realloc(files, capacity * sizeof(char *));
            }
            files[count++] = strdup(name);
        }
    }
    closedir(dir);

    qsort(files, count, sizeof(char *), cmp_str);

    *out_files = files;
    return count;
}

/* ------------------------------------------------------------------ */
/* Accuracy verification                                               */
/* ------------------------------------------------------------------ */

static void verify_detections(zarray_t *detections, const gt_image_t *gt,
                              bench_result_t *result) {
    int n_det = zarray_size(detections);
    int n_gt = gt->num_detections;

    bool *gt_matched = calloc(n_gt > 0 ? n_gt : 1, sizeof(bool));
    bool *det_matched = calloc(n_det > 0 ? n_det : 1, sizeof(bool));

    result->tags_correct = 0;
    result->max_center_err = 0;
    result->max_corner_err = 0;

    for (int di = 0; di < n_det; di++) {
        apriltag_detection_t *det;
        zarray_get(detections, di, &det);

        int best_gi = -1;
        double best_dist = 1e9;

        for (int gi = 0; gi < n_gt; gi++) {
            if (gt_matched[gi]) continue;
            if (gt->detections[gi].tag_id != det->id) continue;
            double dx = det->c[0] - gt->detections[gi].center[0];
            double dy = det->c[1] - gt->detections[gi].center[1];
            double dist = sqrt(dx * dx + dy * dy);
            if (dist < best_dist) {
                best_dist = dist;
                best_gi = gi;
            }
        }

        if (best_gi >= 0) {
            gt_matched[best_gi] = true;
            det_matched[di] = true;
            result->tags_correct++;

            if (best_dist > result->max_center_err)
                result->max_center_err = best_dist;

            for (int c = 0; c < 4; c++) {
                double dx = det->p[c][0] - gt->detections[best_gi].corners[c][0];
                double dy = det->p[c][1] - gt->detections[best_gi].corners[c][1];
                double err = sqrt(dx * dx + dy * dy);
                if (err > result->max_corner_err)
                    result->max_corner_err = err;
            }
        }
    }

    result->tags_missed = 0;
    for (int gi = 0; gi < n_gt; gi++)
        if (!gt_matched[gi]) result->tags_missed++;

    result->tags_extra = 0;
    for (int di = 0; di < n_det; di++)
        if (!det_matched[di]) result->tags_extra++;

    free(gt_matched);
    free(det_matched);
}

/* ------------------------------------------------------------------ */
/* Resolution bin for summary                                          */
/* ------------------------------------------------------------------ */

typedef struct {
    int width, height;
    int count;
    int capacity;
    double *medians;
    int total_gt_tags;
    int total_det_tags;
} res_bin_t;

static res_bin_t *find_or_create_bin(res_bin_t **bins, int *nbins, int *bins_cap,
                                     int w, int h) {
    for (int i = 0; i < *nbins; i++) {
        if ((*bins)[i].width == w && (*bins)[i].height == h)
            return &(*bins)[i];
    }
    if (*nbins >= *bins_cap) {
        *bins_cap *= 2;
        *bins = realloc(*bins, *bins_cap * sizeof(res_bin_t));
    }
    res_bin_t *b = &(*bins)[*nbins];
    memset(b, 0, sizeof(*b));
    b->width = w;
    b->height = h;
    b->capacity = 256;
    b->medians = malloc(b->capacity * sizeof(double));
    (*nbins)++;
    return b;
}

static void bin_add(res_bin_t *b, double median_ms, int gt_tags, int det_tags) {
    if (b->count >= b->capacity) {
        b->capacity *= 2;
        b->medians = realloc(b->medians, b->capacity * sizeof(double));
    }
    b->medians[b->count++] = median_ms;
    b->total_gt_tags += gt_tags;
    b->total_det_tags += det_tags;
}

static int cmp_bin_pixels(const void *a, const void *b) {
    const res_bin_t *ba = a, *bb = b;
    int pa = ba->width * ba->height;
    int pb = bb->width * bb->height;
    return (pa > pb) - (pa < pb);
}

/* ------------------------------------------------------------------ */
/* Summary printing                                                    */
/* ------------------------------------------------------------------ */

static void print_summary(bench_result_t *results, int n,
                          int total_gt, int total_det,
                          int total_correct, int total_missed, int total_extra,
                          int warnings, double global_max_center, double global_max_corner) {
    /* Build resolution bins */
    int nbins = 0, bins_cap = 16;
    res_bin_t *bins = malloc(bins_cap * sizeof(res_bin_t));

    for (int i = 0; i < n; i++) {
        bench_result_t *r = &results[i];
        if (r->image_width == 0) continue; /* skipped image */
        res_bin_t *b = find_or_create_bin(&bins, &nbins, &bins_cap,
                                          r->image_width, r->image_height);
        bin_add(b, r->median_ms, r->gt_num_detections, r->det_num_detections);
    }

    qsort(bins, nbins, sizeof(res_bin_t), cmp_bin_pixels);

    printf("\n=== Summary by Resolution ===\n");
    printf("%-14s %6s %10s %10s %10s %10s %10s\n",
           "Resolution", "Count", "Mean(ms)", "Median(ms)", "P95(ms)", "P99(ms)", "Tags/img");

    /* Collect all medians for the ALL row */
    double *all_medians = malloc(n * sizeof(double));
    int all_count = 0;

    for (int i = 0; i < nbins; i++) {
        res_bin_t *b = &bins[i];
        qsort(b->medians, b->count, sizeof(double), cmp_double);

        double sum = 0;
        for (int j = 0; j < b->count; j++) sum += b->medians[j];
        double mean = sum / b->count;
        double med = percentile(b->medians, b->count, 0.50);
        double p95 = percentile(b->medians, b->count, 0.95);
        double p99 = percentile(b->medians, b->count, 0.99);
        double avg_tags = b->count > 0 ? (double)b->total_det_tags / b->count : 0;

        char label[32];
        snprintf(label, sizeof(label), "%dx%d", b->width, b->height);
        printf("%-14s %6d %10.2f %10.2f %10.2f %10.2f %10.1f\n",
               label, b->count, mean, med, p95, p99, avg_tags);

        memcpy(all_medians + all_count, b->medians, b->count * sizeof(double));
        all_count += b->count;
    }

    /* ALL row */
    qsort(all_medians, all_count, sizeof(double), cmp_double);
    double all_sum = 0;
    for (int i = 0; i < all_count; i++) all_sum += all_medians[i];
    double all_avg_tags = all_count > 0 ? (double)total_det / all_count : 0;

    printf("%-14s %6d %10.2f %10.2f %10.2f %10.2f %10.1f\n",
           "ALL", all_count,
           all_count > 0 ? all_sum / all_count : 0,
           percentile(all_medians, all_count, 0.50),
           percentile(all_medians, all_count, 0.95),
           percentile(all_medians, all_count, 0.99),
           all_avg_tags);

    /* Accuracy */
    printf("\n=== Accuracy ===\n");
    printf("Total GT tags:     %6d\n", total_gt);
    printf("Total detected:    %6d\n", total_det);
    printf("Matched correctly: %6d", total_correct);
    if (total_gt > 0)
        printf(" (%.2f%%)", 100.0 * total_correct / total_gt);
    printf("\n");
    printf("Missed:            %6d\n", total_missed);
    printf("Extra:             %6d\n", total_extra);
    printf("Max center error:  %.4f px\n", global_max_center);
    printf("Max corner error:  %.4f px\n", global_max_corner);
    printf("Accuracy warnings: %6d\n", warnings);

    /* Cleanup */
    for (int i = 0; i < nbins; i++) free(bins[i].medians);
    free(bins);
    free(all_medians);
}

/* ------------------------------------------------------------------ */
/* Summary JSON output                                                 */
/* ------------------------------------------------------------------ */

static void write_summary_json(const char *path, bench_result_t *results, int n,
                               const bench_config_t *cfg,
                               int total_gt, int total_det,
                               int total_correct, int total_missed, int total_extra,
                               double global_max_center, double global_max_corner) {
    cJSON *root = cJSON_CreateObject();

    /* Config */
    cJSON *config = cJSON_CreateObject();
    cJSON_AddNumberToObject(config, "iterations", cfg->iterations);
    cJSON_AddNumberToObject(config, "warmup", cfg->warmup);
    cJSON_AddNumberToObject(config, "nthreads", cfg->nthreads);
    cJSON_AddNumberToObject(config, "quad_decimate", cfg->quad_decimate);
    cJSON_AddNumberToObject(config, "quad_sigma", cfg->quad_sigma);
    cJSON_AddBoolToObject(config, "refine_edges", cfg->refine_edges);
    cJSON_AddNumberToObject(config, "decode_sharpening", cfg->decode_sharpening);
    cJSON_AddNumberToObject(config, "hamming_bits", cfg->hamming_bits);
    cJSON_AddItemToObject(root, "config", config);

    cJSON_AddNumberToObject(root, "total_images", n);

    /* Build resolution bins */
    int nbins = 0, bins_cap = 16;
    res_bin_t *bins = malloc(bins_cap * sizeof(res_bin_t));

    for (int i = 0; i < n; i++) {
        bench_result_t *r = &results[i];
        if (r->image_width == 0) continue;
        res_bin_t *b = find_or_create_bin(&bins, &nbins, &bins_cap,
                                          r->image_width, r->image_height);
        bin_add(b, r->median_ms, r->gt_num_detections, r->det_num_detections);
    }

    qsort(bins, nbins, sizeof(res_bin_t), cmp_bin_pixels);

    cJSON *res_array = cJSON_CreateArray();
    for (int i = 0; i < nbins; i++) {
        res_bin_t *b = &bins[i];
        qsort(b->medians, b->count, sizeof(double), cmp_double);

        double sum = 0;
        for (int j = 0; j < b->count; j++) sum += b->medians[j];

        cJSON *bin = cJSON_CreateObject();
        char label[32];
        snprintf(label, sizeof(label), "%dx%d", b->width, b->height);
        cJSON_AddStringToObject(bin, "resolution", label);
        cJSON_AddNumberToObject(bin, "count", b->count);
        cJSON_AddNumberToObject(bin, "mean_ms", sum / b->count);
        cJSON_AddNumberToObject(bin, "median_ms", percentile(b->medians, b->count, 0.50));
        cJSON_AddNumberToObject(bin, "p95_ms", percentile(b->medians, b->count, 0.95));
        cJSON_AddNumberToObject(bin, "p99_ms", percentile(b->medians, b->count, 0.99));
        cJSON_AddItemToArray(res_array, bin);
    }
    cJSON_AddItemToObject(root, "resolution_bins", res_array);

    /* Accuracy */
    cJSON *acc = cJSON_CreateObject();
    cJSON_AddNumberToObject(acc, "total_gt_tags", total_gt);
    cJSON_AddNumberToObject(acc, "total_detected", total_det);
    cJSON_AddNumberToObject(acc, "matched", total_correct);
    cJSON_AddNumberToObject(acc, "missed", total_missed);
    cJSON_AddNumberToObject(acc, "extra", total_extra);
    cJSON_AddNumberToObject(acc, "max_center_error_px", global_max_center);
    cJSON_AddNumberToObject(acc, "max_corner_error_px", global_max_corner);
    cJSON_AddItemToObject(root, "accuracy", acc);

    char *str = cJSON_Print(root);
    FILE *f = fopen(path, "w");
    if (f) {
        fputs(str, f);
        fputc('\n', f);
        fclose(f);
    }
    free(str);
    cJSON_Delete(root);

    for (int i = 0; i < nbins; i++) free(bins[i].medians);
    free(bins);
}

/* ------------------------------------------------------------------ */
/* CLI parsing                                                         */
/* ------------------------------------------------------------------ */

static void print_usage(const char *progname) {
    printf("Usage: %s [options]\n\n", progname);
    printf("Options:\n");
    printf("  -d, --data-dir <path>       Dataset root (default: ../data)\n");
    printf("  -n, --iterations <N>        Iterations per image (default: 10)\n");
    printf("  -w, --warmup <N>            Warmup iterations to discard (default: 1)\n");
    printf("  -t, --threads <N>           Detector threads (default: 4)\n");
    printf("  -x, --decimate <F>          Quad decimation factor (default: 1.0)\n");
    printf("  -b, --blur <F>              Gaussian blur sigma (default: 0.0)\n");
    printf("      --refine-edges <0|1>    Refine tag edges (default: 1)\n");
    printf("      --sharpening <F>        Decode sharpening (default: 0.25)\n");
    printf("      --hamming <N>           Max hamming bit errors (default: 2)\n");
    printf("  -o, --output <path>         CSV output file (default: results.csv)\n");
    printf("  -s, --summary <path>        Summary JSON file (default: summary.json)\n");
    printf("  -q, --quiet                 Suppress per-image progress\n");
    printf("  -l, --limit <N>             Process only first N images (0=all)\n");
    printf("      --no-verify             Skip accuracy verification\n");
    printf("  -h, --help                  Show this help\n");
}

enum {
    OPT_REFINE_EDGES = 256,
    OPT_SHARPENING,
    OPT_HAMMING,
    OPT_NO_VERIFY,
};

static struct option long_options[] = {
    {"data-dir",     required_argument, 0, 'd'},
    {"iterations",   required_argument, 0, 'n'},
    {"warmup",       required_argument, 0, 'w'},
    {"threads",      required_argument, 0, 't'},
    {"decimate",     required_argument, 0, 'x'},
    {"blur",         required_argument, 0, 'b'},
    {"refine-edges", required_argument, 0, OPT_REFINE_EDGES},
    {"sharpening",   required_argument, 0, OPT_SHARPENING},
    {"hamming",      required_argument, 0, OPT_HAMMING},
    {"output",       required_argument, 0, 'o'},
    {"summary",      required_argument, 0, 's'},
    {"quiet",        no_argument,       0, 'q'},
    {"limit",        required_argument, 0, 'l'},
    {"no-verify",    no_argument,       0, OPT_NO_VERIFY},
    {"help",         no_argument,       0, 'h'},
    {0, 0, 0, 0}
};

static bench_config_t parse_args(int argc, char *argv[]) {
    bench_config_t cfg = {
        .data_dir = "../data",
        .iterations = 10,
        .warmup = 1,
        .nthreads = 4,
        .quad_decimate = 1.0f,
        .quad_sigma = 0.0f,
        .refine_edges = 1,
        .decode_sharpening = 0.25,
        .hamming_bits = 2,
        .output_csv = "results.csv",
        .summary_json = "summary.json",
        .quiet = 0,
        .limit = 0,
        .verify = 1,
    };

    int c;
    while ((c = getopt_long(argc, argv, "d:n:w:t:x:b:o:s:ql:h", long_options, NULL)) != -1) {
        switch (c) {
            case 'd': cfg.data_dir = optarg; break;
            case 'n': cfg.iterations = atoi(optarg); break;
            case 'w': cfg.warmup = atoi(optarg); break;
            case 't': cfg.nthreads = atoi(optarg); break;
            case 'x': cfg.quad_decimate = (float)atof(optarg); break;
            case 'b': cfg.quad_sigma = (float)atof(optarg); break;
            case OPT_REFINE_EDGES: cfg.refine_edges = atoi(optarg); break;
            case OPT_SHARPENING: cfg.decode_sharpening = atof(optarg); break;
            case OPT_HAMMING: cfg.hamming_bits = atoi(optarg); break;
            case 'o': cfg.output_csv = optarg; break;
            case 's': cfg.summary_json = optarg; break;
            case 'q': cfg.quiet = 1; break;
            case 'l': cfg.limit = atoi(optarg); break;
            case OPT_NO_VERIFY: cfg.verify = 0; break;
            case 'h': print_usage(argv[0]); exit(0);
            default: print_usage(argv[0]); exit(1);
        }
    }

    if (cfg.warmup >= cfg.iterations) {
        fprintf(stderr, "ERROR: warmup (%d) must be less than iterations (%d)\n",
                cfg.warmup, cfg.iterations);
        exit(1);
    }

    return cfg;
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */

int main(int argc, char *argv[]) {
    bench_config_t cfg = parse_args(argc, argv);

    /* Discover images */
    char **image_files;
    int num_images = discover_images(cfg.data_dir, &image_files);
    if (num_images == 0) {
        fprintf(stderr, "ERROR: No PNG images found in %s/images/\n", cfg.data_dir);
        return 1;
    }
    if (cfg.limit > 0 && cfg.limit < num_images)
        num_images = cfg.limit;

    /* Create detector */
    apriltag_family_t *tf = tag36h11_create();
    apriltag_detector_t *td = apriltag_detector_create();
    apriltag_detector_add_family_bits(td, tf, cfg.hamming_bits);

    if (errno == EINVAL) {
        fprintf(stderr, "ERROR: hamming parameter out of range\n");
        return 1;
    }
    if (errno == ENOMEM) {
        fprintf(stderr, "ERROR: insufficient memory for tag family decoder\n");
        return 1;
    }

    td->quad_decimate = cfg.quad_decimate;
    td->quad_sigma = cfg.quad_sigma;
    td->nthreads = cfg.nthreads;
    td->debug = 0;
    td->refine_edges = cfg.refine_edges;
    td->decode_sharpening = cfg.decode_sharpening;

    int measured_iters = cfg.iterations - cfg.warmup;

    /* Print header */
    printf("Apriltag Benchmark\n");
    printf("  Images:     %d\n", num_images);
    printf("  Iterations: %d (%d warmup + %d measured)\n",
           cfg.iterations, cfg.warmup, measured_iters);
    printf("  Threads:    %d\n", cfg.nthreads);
    printf("  Config:     decimate=%.1f sigma=%.1f refine=%d sharpening=%.2f hamming=%d\n",
           td->quad_decimate, td->quad_sigma, td->refine_edges,
           td->decode_sharpening, cfg.hamming_bits);
    printf("\n");

    /* Open CSV */
    FILE *csv = fopen(cfg.output_csv, "w");
    if (!csv) {
        fprintf(stderr, "ERROR: Cannot open output CSV: %s\n", cfg.output_csv);
        return 1;
    }
    fprintf(csv, "filename,width,height,resolution,gt_tags,det_tags,"
                 "tags_correct,tags_missed,tags_extra,"
                 "max_center_err_px,max_corner_err_px,"
                 "min_ms,max_ms,mean_ms,median_ms,stddev_ms\n");

    /* Allocate results */
    bench_result_t *results = calloc(num_images, sizeof(bench_result_t));

    int total_gt = 0, total_det = 0;
    int total_correct = 0, total_missed = 0, total_extra = 0;
    int accuracy_warnings = 0;
    double global_max_center = 0, global_max_corner = 0;
    int processed = 0;

    /* Main loop */
    for (int i = 0; i < num_images; i++) {
        const char *img_filename = image_files[i];
        bench_result_t *r = &results[i];

        /* Build paths */
        char img_path[2048], gt_path[2048];
        snprintf(img_path, sizeof(img_path), "%s/images/%s", cfg.data_dir, img_filename);

        char stem[512];
        strncpy(stem, img_filename, sizeof(stem) - 1);
        stem[sizeof(stem) - 1] = '\0';
        char *dot = strrchr(stem, '.');
        if (dot) *dot = '\0';
        snprintf(gt_path, sizeof(gt_path), "%s/detections/%s.json", cfg.data_dir, stem);

        /* Load ground truth */
        gt_image_t gt = {0};
        int gt_ok = parse_ground_truth(gt_path, &gt);
        if (gt_ok != 0 && cfg.verify) {
            if (!cfg.quiet)
                fprintf(stderr, "WARN: No ground truth for %s, skipping accuracy check\n",
                        img_filename);
        }

        /* Load image */
        image_u8_t *im = load_png_as_image_u8(img_path);
        if (!im) {
            fprintf(stderr, "ERROR: Failed to load %s\n", img_path);
            free(gt.detections);
            continue;
        }

        strncpy(r->filename, img_filename, sizeof(r->filename) - 1);
        r->image_width = im->width;
        r->image_height = im->height;
        r->gt_num_detections = gt.num_detections;

        /* Warmup */
        for (int w = 0; w < cfg.warmup; w++) {
            zarray_t *dets = apriltag_detector_detect(td, im);
            apriltag_detections_destroy(dets);
        }

        /* Measured iterations */
        double *times = malloc(measured_iters * sizeof(double));
        zarray_t *last_detections = NULL;

        for (int iter = 0; iter < measured_iters; iter++) {
            double t0 = get_time_ms();
            zarray_t *dets = apriltag_detector_detect(td, im);
            double t1 = get_time_ms();
            times[iter] = t1 - t0;

            if (iter < measured_iters - 1) {
                apriltag_detections_destroy(dets);
            } else {
                last_detections = dets;
            }
        }

        r->det_num_detections = zarray_size(last_detections);

        /* Timing stats */
        timing_stats_t stats = compute_stats(times, measured_iters);
        r->min_ms = stats.min;
        r->max_ms = stats.max;
        r->mean_ms = stats.mean;
        r->median_ms = stats.median;
        r->stddev_ms = stats.stddev;

        /* Accuracy */
        if (cfg.verify && gt_ok == 0) {
            verify_detections(last_detections, &gt, r);

            if (r->max_center_err > 1.0 || r->tags_missed > 0 || r->tags_extra > 0) {
                accuracy_warnings++;
                if (!cfg.quiet) {
                    fprintf(stderr, "ACCURACY: %s  correct=%d missed=%d extra=%d "
                            "center_err=%.4f corner_err=%.4f\n",
                            img_filename, r->tags_correct, r->tags_missed,
                            r->tags_extra, r->max_center_err, r->max_corner_err);
                }
            }

            if (r->max_center_err > global_max_center)
                global_max_center = r->max_center_err;
            if (r->max_corner_err > global_max_corner)
                global_max_corner = r->max_corner_err;
        } else {
            /* No GT or verification disabled */
            r->tags_correct = r->det_num_detections;
            r->tags_missed = 0;
            r->tags_extra = 0;
            r->max_center_err = 0;
            r->max_corner_err = 0;
        }

        total_gt += r->gt_num_detections;
        total_det += r->det_num_detections;
        total_correct += r->tags_correct;
        total_missed += r->tags_missed;
        total_extra += r->tags_extra;

        /* CSV row */
        fprintf(csv, "%s,%d,%d,%dx%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                r->filename, r->image_width, r->image_height,
                r->image_width, r->image_height,
                r->gt_num_detections, r->det_num_detections,
                r->tags_correct, r->tags_missed, r->tags_extra,
                r->max_center_err, r->max_corner_err,
                r->min_ms, r->max_ms, r->mean_ms, r->median_ms, r->stddev_ms);

        /* Progress */
        if (!cfg.quiet) {
            printf("[%4d/%4d] %-70.70s %4dx%-4d  tags=%2d  median=%.2f ms\n",
                   i + 1, num_images, img_filename,
                   im->width, im->height,
                   r->det_num_detections, r->median_ms);
        }

        processed++;
        apriltag_detections_destroy(last_detections);
        image_u8_destroy(im);
        free(gt.detections);
        free(times);
    }
    fclose(csv);

    /* Summary */
    print_summary(results, num_images,
                  total_gt, total_det, total_correct, total_missed, total_extra,
                  accuracy_warnings, global_max_center, global_max_corner);

    printf("\nProcessed %d images. Results written to %s\n", processed, cfg.output_csv);

    /* Summary JSON */
    write_summary_json(cfg.summary_json, results, num_images, &cfg,
                       total_gt, total_det, total_correct, total_missed, total_extra,
                       global_max_center, global_max_corner);
    printf("Summary written to %s\n", cfg.summary_json);

    /* Cleanup */
    free(results);
    for (int i = 0; i < num_images; i++) free(image_files[i]);
    free(image_files);
    apriltag_detector_destroy(td);
    tag36h11_destroy(tf);

    return 0;
}
