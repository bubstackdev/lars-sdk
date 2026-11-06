/* The C ABI end to end on a pack, as a binding uses it: sync and async answers, errors,
 * calibrators, concurrent callers, cancellation and close. Run by tests/c_smoke.rs:
 *     smoke <pack-dir> <calibration.jsonl>
 * Exits non-zero on the first failed check. */
#define _DEFAULT_SOURCE /* usleep under -std=c11 on glibc */
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "lars.h"

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAILED %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); exit(1); } } while (0)

static const char *REQUEST =
    "{\"state\": \"I was charged twice. Refund me!\", \"questions\": {"
    "\"refund\": {\"type\": \"noul\", \"instructions\": \"The customer asks for money back\"},"
    "\"team\": {\"type\": \"choice\", \"instructions\": \"Which team?\","
    "\"criteria\": {\"billing\": \"Charges\", \"technical\": \"Bugs\"}}}}";

static atomic_int replies, cancelled, failures;

static void on_reply(void *user, uint64_t id, const char *result, const char *error) {
    (void)user;
    if (result && strstr(result, "\"answers\"") && !error) {
        atomic_fetch_add(&replies, 1);
    } else if (error && strstr(error, "\"cancelled\"")) {
        atomic_fetch_add(&cancelled, 1);
    } else {
        fprintf(stderr, "unexpected reply to %llu: %s %s\n", (unsigned long long)id, result ? result : "-", error ? error : "-");
        atomic_fetch_add(&failures, 1);
    }
}

static atomic_int parse_errors;

static void on_parse_error(void *user, uint64_t id, const char *result, const char *error) {
    (void)user;
    if (id == 0 && !result && error && strstr(error, "invalid_request")) atomic_fetch_add(&parse_errors, 1);
}

static void wait_for(atomic_int *count, int target) {
    for (int i = 0; i < 2000 && atomic_load(count) < target; i++) usleep(5000);
    CHECK(atomic_load(count) >= target, "waited for %d replies, got %d", target, atomic_load(count));
}

static void *caller(void *engine) {
    for (int i = 0; i < 50; i++) {
        char *err = NULL;
        char *answer = lars_evaluate(engine, REQUEST, &err);
        CHECK(answer && !err, "threaded evaluate: %s", err ? err : "?");
        lars_string_free(answer);
    }
    return NULL;
}

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    CHECK(f, "cannot read %s", path);
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *text = malloc((size_t)n + 1);
    CHECK(fread(text, 1, (size_t)n, f) == (size_t)n, "short read");
    text[n] = 0;
    fclose(f);
    return text;
}

int main(int argc, char **argv) {
    CHECK(argc == 3, "usage: smoke <pack-dir> <calibration.jsonl>");
    printf("lars %s\n", lars_version());

    char *err = NULL;
    CHECK(!lars_open("./no/such/pack", NULL, &err) && err && strstr(err, "pack_error"), "bad pack: %s", err ? err : "?");
    lars_string_free(err);
    err = NULL;
    lars_engine *engine = lars_open(argv[1], "{\"threads\": 2}", &err);
    CHECK(engine && !err, "open: %s", err ? err : "?");

    CHECK(!lars_evaluate_image_v1(engine, REQUEST, NULL, 0, &err) && strstr(err, "invalid_request"), "NULL image");
    lars_string_free(err);
    err = NULL;
    const uint8_t image[] = {1, 2, 3};
    CHECK(!lars_evaluate_image_v1(engine, REQUEST, image, sizeof(image), &err) && strstr(err, "does not support images"), "text pack rejects image");
    lars_string_free(err);
    err = NULL;
    int before_images = atomic_load(&parse_errors);
    CHECK(lars_evaluate_image_async_v1(engine, REQUEST, image, sizeof(image), on_parse_error, NULL) == 0, "unsupported image replies immediately");
    CHECK(atomic_load(&parse_errors) == before_images + 1, "image callback ran exactly once");
    atomic_store(&parse_errors, before_images);

    char *answer = lars_evaluate(engine, REQUEST, &err);
    CHECK(answer && strstr(answer, "\"refund\""), "evaluate: %s", err ? err : "?");
    lars_string_free(answer);
    CHECK(!lars_evaluate(engine, "{\"state\": 1}", &err) && strstr(err, "invalid_request"), "bad request");
    lars_string_free(err);
    err = NULL;
    CHECK(!lars_evaluate(NULL, REQUEST, &err) && strstr(err, "NULL"), "NULL engine");
    lars_string_free(err);
    err = NULL;

    for (int i = 0; i < 100; i++) CHECK(lars_evaluate_async(engine, REQUEST, on_reply, NULL) > 0, "queued");
    wait_for(&replies, 100);
    CHECK(lars_evaluate_async(engine, "not json", on_parse_error, NULL) == 0 && atomic_load(&parse_errors) == 1,
          "a request that does not parse is replied to at once, with id 0");

    char *examples = read_file(argv[2]);
    char *fitted = lars_calibrate(engine, "triage", examples, 0.05, &err);
    CHECK(fitted && strstr(fitted, "\"calibrator\"") && strstr(fitted, "\"certified\""), "calibrate: %s", err ? err : "?");
    lars_string_free(fitted);
    free(examples);
    char *saved = lars_calibrators(engine);
    CHECK(saved && strstr(saved, "\"triage\""), "calibrators");
    lars_string_free(saved);
    CHECK(lars_set_calibrator(engine, "x", "{\"temperatures\": {}}", &err) == -1 && strstr(err, "records no pack"),
          "unbound calibrator refused");
    lars_string_free(err);
    err = NULL;

    pthread_t threads[4];
    for (int i = 0; i < 4; i++) pthread_create(&threads[i], NULL, caller, engine);
    for (int i = 0; i < 4; i++) pthread_join(threads[i], NULL);

    char *status = lars_status(engine);
    CHECK(status && strstr(status, "\"queued\":0"), "status: %s", status ? status : "?");
    lars_string_free(status);

    atomic_store(&replies, 0);
    uint64_t ids[300];
    for (int i = 0; i < 300; i++) ids[i] = lars_evaluate_async(engine, REQUEST, on_reply, NULL);
    for (int i = 150; i < 300; i++) lars_cancel(engine, ids[i]);
    lars_close(engine); /* cancels what is still queued; every callback still runs */
    for (int i = 0; i < 2000 && atomic_load(&replies) + atomic_load(&cancelled) < 300; i++) usleep(5000);
    CHECK(atomic_load(&replies) + atomic_load(&cancelled) == 300 && atomic_load(&failures) == 0,
          "300 callbacks: %d answered, %d cancelled, %d failed", atomic_load(&replies), atomic_load(&cancelled),
          atomic_load(&failures));
    printf("ok: %d answered and %d cancelled after close\n", atomic_load(&replies), atomic_load(&cancelled));
    usleep(100000); /* let the worker thread finish and free the model before the leak check */
    return 0;
}
