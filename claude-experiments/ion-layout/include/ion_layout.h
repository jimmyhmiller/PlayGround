/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * Part of an MPL-2.0 Graphviz layout plugin derived from iongraph.
 * See NOTICE.md for attribution. */

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct IonPoint {
    double x;
    double y;
} IonPoint;

/* Per-edge route descriptor. `offset` indexes into the shared points buffer
 * returned via route_points; the route occupies point_count consecutive
 * points (cubic bezier control points, 3k+1 of them). point_count == 0
 * means the edge has no drawable route. */
typedef struct IonRoute {
    size_t offset;
    size_t point_count;
    IonPoint arrow_tip;
} IonRoute;

/* Flow direction (Graphviz rankdir). */
enum IonOrientation {
    ION_TOP_TO_BOTTOM = 0,
    ION_LEFT_TO_RIGHT = 1,
    ION_BOTTOM_TO_TOP = 2,
    ION_RIGHT_TO_LEFT = 3,
};

/* widths/heights are in-out: caller fills with label sizes, the layout
 * writes back the clamped cell sizes it used.
 *
 * routes receives one descriptor per input edge, in input order. All route
 * points live in a single buffer returned through route_points /
 * route_points_len; the caller must copy what it needs and then release the
 * buffer with ion_layout_free_points. */
int ion_layout_compute(
    size_t node_count,
    double *widths,
    double *heights,
    size_t edge_count,
    const size_t *tails,
    const size_t *heads,
    const int *loop_depths,
    const unsigned char *loop_headers,
    const unsigned char *backedges,
    unsigned int orientation,
    IonPoint *positions,
    IonRoute *routes,
    IonPoint **route_points,
    size_t *route_points_len,
    double *graph_width,
    double *graph_height
);

void ion_layout_free_points(IonPoint *points, size_t len);

#ifdef __cplusplus
}
#endif
