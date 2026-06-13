/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * Part of an MPL-2.0 Graphviz layout plugin derived from iongraph.
 * See NOTICE.md for attribution. */

#include <graphviz/cgraph.h>
#include <graphviz/geom.h>
#include <graphviz/gvplugin.h>
#include <graphviz/gvplugin_layout.h>
#include <graphviz/types.h>

#include <stdlib.h>\n#include <stdio.h>
#include <string.h>

#include "ion_layout.h"

#ifndef POINTS_PER_INCH
#define POINTS_PER_INCH 72.0
#endif

// These helpers are exported by libgvc but are not currently declared in the
// public plugin headers Graphviz installs.
extern void common_init_node(node_t *n);
extern int common_init_edge(edge_t *e);
extern void gv_nodesize(node_t *n, int flip);
extern void gv_cleanup_node(node_t *n);
extern void gv_cleanup_edge(edge_t *e);
extern void dotneato_postprocess(graph_t *g);
extern void setEdgeType(graph_t *g, int dflt);
extern void arrow_flags(edge_t *e, uint32_t *sflag, uint32_t *eflag);
/* Full graph-level init/teardown (GD_drawing, label, charset...). Renderers
 * that store xdot attributes (-Txdot, -Tjson) crash without it. */
extern void graph_init(graph_t *g, _Bool use_rankdir);
extern void graph_cleanup(graph_t *g);
/* libgvc's "last finished layout phase". attach_attrs_and_arrows() only
 * scans splines (and declares the _hdraw_/_tdraw_ xdot attributes) when
 * State >= GVSPLINES; without this, -Txdot/-Tjson agxset a NULL symbol and
 * crash. Layout engines are responsible for advancing it. */
extern int State;
#define ION_GVSPLINES 1 /* GVSPLINES from lib/common/const.h */

#define ET_SPLINE (5 << 1)

static pointf flip_point(IonPoint p, double graph_height) {
    pointf out = {p.x, graph_height - p.y};
    return out;
}

/* Already-placed edge label boxes, so colliding labels (e.g. on parallel
 * edges that merge into one line) can be staggered vertically. */
typedef struct LabelBox { double x, y, w, h; } LabelBox;

static void set_edge_route(edge_t *e, IonRoute route, const IonPoint *route_points, double graph_height, double *max_label_x,
                           LabelBox *placed, size_t *placed_count) {
    size_t point_count = route.point_count;
    /* A drawable route is 3k+1 cubic bezier control points. Anything else
     * means the layout produced no route for this edge; skip rather than
     * render garbage. */
    if (point_count < 4 || (point_count - 1) % 3 != 0) return;

    splines *spl = calloc(1, sizeof(splines));
    bezier *bez = calloc(1, sizeof(bezier));
    pointf *points = calloc(point_count, sizeof(pointf));
    if (!spl || !bez || !points) {
        free(spl);
        free(bez);
        free(points);
        return;
    }

    for (size_t i = 0; i < point_count; i++) points[i] = flip_point(route_points[route.offset + i], graph_height);

    uint32_t sflag = 0;
    uint32_t eflag = 0;
    arrow_flags(e, &sflag, &eflag);

    bez->list = points;
    bez->size = point_count;
    bez->sflag = sflag;
    bez->eflag = eflag;
    bez->sp = points[0];
    bez->ep = points[point_count - 1];
    if (eflag) bez->ep = flip_point(route.arrow_tip, graph_height);

    spl->list = bez;
    spl->size = 1;
    spl->bb.LL.x = spl->bb.UR.x = points[0].x;
    spl->bb.LL.y = spl->bb.UR.y = points[0].y;
    for (size_t i = 1; i < point_count; i++) {
        if (points[i].x < spl->bb.LL.x) spl->bb.LL.x = points[i].x;
        if (points[i].y < spl->bb.LL.y) spl->bb.LL.y = points[i].y;
        if (points[i].x > spl->bb.UR.x) spl->bb.UR.x = points[i].x;
        if (points[i].y > spl->bb.UR.y) spl->bb.UR.y = points[i].y;
    }

    ED_spl(e) = spl;

    textlabel_t *label = ED_label(e);
    if (label) {
        /* Park the label beside the route's midpoint rather than centered on
         * the line: routes are mostly vertical, so shifting the label's
         * center right by half its width keeps the text clear of the edge. */
        label->pos = points[point_count / 2];
        label->pos.x += label->dimen.x / 2.0 + 4.0;
        /* Parallel edges can merge into one line, putting their labels on
         * the same midpoint; stagger collisions downward. */
        for (size_t i = 0; i < *placed_count; i++) {
            double dx = label->pos.x - placed[i].x;
            double dy = label->pos.y - placed[i].y;
            if (dx < 0) dx = -dx;
            if (dy < 0) dy = -dy;
            if (dx < (label->dimen.x + placed[i].w) / 2.0 && dy < (label->dimen.y + placed[i].h) / 2.0) {
                label->pos.y = placed[i].y - (placed[i].h + label->dimen.y) / 2.0 - 2.0;
                i = (size_t)-1; /* restart scan after moving */
            }
        }
        placed[*placed_count] = (LabelBox){label->pos.x, label->pos.y, label->dimen.x, label->dimen.y};
        (*placed_count)++;
        label->set = true;
        double right = label->pos.x + label->dimen.x / 2.0;
        if (right > *max_label_x) *max_label_x = right;
    }
}

static int node_index(node_t **nodes, size_t count, node_t *needle) {
    for (size_t i = 0; i < count; i++) {
        if (nodes[i] == needle) return (int)i;
    }
    return -1;
}

static void ion_layout(graph_t *g) {
    graph_init(g, 0);
    size_t node_count = 0;
    size_t edge_count = 0;

    for (node_t *n = agfstnode(g); n; n = agnxtnode(g, n)) node_count++;
    for (node_t *n = agfstnode(g); n; n = agnxtnode(g, n)) {
        for (edge_t *e = agfstout(g, n); e; e = agnxtout(g, e)) edge_count++;
    }

    node_t **nodes = calloc(node_count ? node_count : 1, sizeof(node_t *));
    double *widths = calloc(node_count ? node_count : 1, sizeof(double));
    double *heights = calloc(node_count ? node_count : 1, sizeof(double));
    int *loop_depths = calloc(node_count ? node_count : 1, sizeof(int));
    unsigned char *loop_headers = calloc(node_count ? node_count : 1, sizeof(unsigned char));
    unsigned char *backedges = calloc(node_count ? node_count : 1, sizeof(unsigned char));
    IonPoint *positions = calloc(node_count ? node_count : 1, sizeof(IonPoint));
    IonRoute *routes = calloc(edge_count ? edge_count : 1, sizeof(IonRoute));
    size_t *tails = calloc(edge_count ? edge_count : 1, sizeof(size_t));
    size_t *heads = calloc(edge_count ? edge_count : 1, sizeof(size_t));
    edge_t **ordered_edges = calloc(edge_count ? edge_count : 1, sizeof(edge_t *));

    if (!nodes || !widths || !heights || !loop_depths || !loop_headers || !backedges || !positions || !routes || !tails || !heads || !ordered_edges) goto done;

    size_t i = 0;
    for (node_t *n = agfstnode(g); n; n = agnxtnode(g, n), i++) {
        agbindrec(n, "Agnodeinfo_t", sizeof(Agnodeinfo_t), true);
        nodes[i] = n;
        common_init_node(n);
        gv_nodesize(n, 0);
        widths[i] = ND_width(n) * POINTS_PER_INCH;
        heights[i] = ND_height(n) * POINTS_PER_INCH;
        char *depth = agget(n, "ion_loop_depth");
        loop_depths[i] = depth ? atoi(depth) : 0;
        char *header = agget(n, "ion_loop_header");
        char *backedge = agget(n, "ion_backedge");
        loop_headers[i] = header && (strcmp(header, "true") == 0 || strcmp(header, "1") == 0);
        backedges[i] = backedge && (strcmp(backedge, "true") == 0 || strcmp(backedge, "1") == 0);
    }

    // agfstout iterates out-edges in LIFO (reverse-declaration) order. We want
    // DOT-declaration order so successor port 0 matches the first edge written
    // cgraph's agfstout yields out-edges in DOT-declaration order, so keep
    // them as-is — port 0 = first declared edge.
    size_t edge_i = 0;
    for (node_t *n = agfstnode(g); n; n = agnxtnode(g, n)) {
        for (edge_t *e = agfstout(g, n); e; e = agnxtout(g, e)) {
            agbindrec(e, "Agedgeinfo_t", sizeof(Agedgeinfo_t), true);
            common_init_edge(e);
            int tail = node_index(nodes, node_count, agtail(e));
            int head = node_index(nodes, node_count, aghead(e));
            if (tail >= 0 && head >= 0) {
                tails[edge_i] = (size_t)tail;
                heads[edge_i] = (size_t)head;
                ordered_edges[edge_i] = e;
                edge_i++;
            }
        }
    }

    unsigned int orientation = ION_TOP_TO_BOTTOM;
    char *rankdir = agget(g, "rankdir");
    if (rankdir) {
        if (strcmp(rankdir, "LR") == 0) orientation = ION_LEFT_TO_RIGHT;
        else if (strcmp(rankdir, "BT") == 0) orientation = ION_BOTTOM_TO_TOP;
        else if (strcmp(rankdir, "RL") == 0) orientation = ION_RIGHT_TO_LEFT;
    }

    if (getenv("ION_DUMP_INPUT")) {
        for (size_t k = 0; k < node_count; k++)
            fprintf(stderr, "node %.17g %.17g %d %d %d\n", widths[k], heights[k], loop_depths[k], loop_headers[k], backedges[k]);
        for (size_t k = 0; k < edge_i; k++)
            fprintf(stderr, "edge %zu %zu\n", tails[k], heads[k]);
    }

    double graph_width = 0.0;
    double graph_height = 0.0;
    IonPoint *route_points = NULL;
    size_t route_points_len = 0;
    if (ion_layout_compute(node_count, widths, heights, edge_i, tails, heads, loop_depths, loop_headers, backedges, orientation, positions, routes, &route_points, &route_points_len, &graph_width, &graph_height) != 0) {
        goto done;
    }

    setEdgeType(g, ET_SPLINE);
    double *pre_node_y = calloc(node_count ? node_count : 1, sizeof(double));
    if (!pre_node_y) { ion_layout_free_points(route_points, route_points_len); goto done; }
    for (i = 0; i < node_count; i++) {
        node_t *n = nodes[i];
        // widths[i]/heights[i] are now the EXPANDED cell sizes written back
        // by ion_layout_compute (e.g. nodes widened to fit their output
        // ports). Setting ND_width alone is not enough: the node's shape
        // POLYGON was already built by common_init_node from the label size,
        // and renderers draw that polygon. Re-initialize the shape with the
        // layout's size as a minimum via the width/height attributes.
        double grow_w = widths[i] - ND_width(n) * POINTS_PER_INCH;
        double grow_h = heights[i] - ND_height(n) * POINTS_PER_INCH;
        if (grow_w > 0.01 || grow_h > 0.01) {
            /* poly_init sizes non-regular shapes from ND_width/ND_height
             * (maxed against the label), so set those first, then rebuild
             * the shape polygon. */
            ND_width(n) = widths[i] / POINTS_PER_INCH;
            ND_height(n) = heights[i] / POINTS_PER_INCH;
            if (ND_shape(n) && ND_shape(n)->fns) {
                if (ND_shape(n)->fns->freefn) ND_shape(n)->fns->freefn(n);
                if (ND_shape(n)->fns->initfn) ND_shape(n)->fns->initfn(n);
            }
        }
        ND_pos(n) = calloc(2, sizeof(double));
        if (!ND_pos(n)) { free(pre_node_y); ion_layout_free_points(route_points, route_points_len); goto done; }
        ND_width(n) = widths[i] / POINTS_PER_INCH;
        ND_height(n) = heights[i] / POINTS_PER_INCH;
        ND_lw(n) = widths[i] / 2.0;
        ND_rw(n) = widths[i] / 2.0;
        ND_ht(n) = heights[i];
        ND_coord(n).x = positions[i].x;
        ND_coord(n).y = graph_height - positions[i].y;
        ND_pos(n)[0] = ND_coord(n).x / POINTS_PER_INCH;
        ND_pos(n)[1] = ND_coord(n).y / POINTS_PER_INCH;
        ND_bb(n).LL.x = ND_coord(n).x - widths[i] / 2.0;
        ND_bb(n).LL.y = ND_coord(n).y - heights[i] / 2.0;
        ND_bb(n).UR.x = ND_coord(n).x + widths[i] / 2.0;
        ND_bb(n).UR.y = ND_coord(n).y + heights[i] / 2.0;
        pre_node_y[i] = ND_coord(n).y;
    }

    GD_bb(g).LL.x = 0.0;
    GD_bb(g).LL.y = 0.0;
    GD_bb(g).UR.x = graph_width;
    GD_bb(g).UR.y = graph_height;

    double max_label_x = 0.0;
    LabelBox *placed_labels = calloc(edge_i ? edge_i : 1, sizeof(LabelBox));
    size_t placed_count = 0;
    for (size_t i2 = 0; i2 < edge_i; i2++) {
        set_edge_route(ordered_edges[i2], routes[i2], route_points, graph_height, &max_label_x,
                       placed_labels ? placed_labels : (LabelBox[]){{0}}, &placed_count);
        if (!placed_labels) placed_count = 0;
    }
    free(placed_labels);
    ion_layout_free_points(route_points, route_points_len);
    route_points = NULL;
    /* Edge labels sit beside their routes and can poke past the layout's
     * width; grow the bounding box so they aren't clipped. */
    if (max_label_x + 8.0 > GD_bb(g).UR.x) GD_bb(g).UR.x = max_label_x + 8.0;

    State = ION_GVSPLINES;

    dotneato_postprocess(g);

    // dotneato_postprocess may shift node ND_coord.y (e.g. to make room for a
    // graph-level label that extends the bounding box). It does NOT always
    // shift nodes by the same amount the bbox grows — when the extra space is
    // absorbed by extending LL.y, nodes stay put. Measure the actual shift on
    // a real node and apply the same delta to spline points. Using bbox-delta
    // here would over-shift and leave arrows floating away from nodes.
    double dy = 0.0;
    for (i = 0; i < node_count; i++) {
        double d = ND_coord(nodes[i]).y - pre_node_y[i];
        if (d != 0.0) { dy = d; break; }
    }
    free(pre_node_y);
    if (dy != 0.0) {
        for (node_t *n = agfstnode(g); n; n = agnxtnode(g, n)) {
            for (edge_t *e = agfstout(g, n); e; e = agnxtout(g, e)) {
                splines *spl = ED_spl(e);
                if (!spl) continue;
                for (size_t bi = 0; bi < spl->size; bi++) {
                    bezier *bez = &spl->list[bi];
                    for (size_t pi = 0; pi < bez->size; pi++) bez->list[pi].y += dy;
                    bez->sp.y += dy;
                    bez->ep.y += dy;
                }
                spl->bb.LL.y += dy;
                spl->bb.UR.y += dy;
                textlabel_t *label = ED_label(e);
                if (label && label->set) label->pos.y += dy;
            }
        }
    }

done:
    free(ordered_edges);
    free(nodes);
    free(widths);
    free(heights);
    free(loop_depths);
    free(loop_headers);
    free(backedges);
    free(positions);
    free(routes);
    free(tails);
    free(heads);
}

static void ion_cleanup(graph_t *g) {
    for (node_t *n = agfstnode(g); n; n = agnxtnode(g, n)) {
        for (edge_t *e = agfstout(g, n); e; e = agnxtout(g, e)) gv_cleanup_edge(e);
        gv_cleanup_node(n);
    }
    graph_cleanup(g);
}

static gvlayout_engine_t ion_engine = {
    ion_layout,
    ion_cleanup,
};

static gvlayout_features_t ion_features = {0};

static gvplugin_installed_t ion_layout_types[] = {
    {0, "ion", 0, &ion_engine, &ion_features},
    {0, NULL, 0, NULL, NULL},
};

static gvplugin_api_t ion_apis[] = {
    {API_layout, ion_layout_types},
    {(api_t)0, NULL},
};

gvplugin_library_t gvplugin_ion_LTX_library = {
    "ion",
    ion_apis,
};
