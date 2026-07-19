/*
 * Copyright 2026 Jimmy Miller
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0. See LICENSE at the
 * repository root, and THIRD_PARTY_NOTICES.md for the upstream work this
 * derives from.
 */

/*
 * A reference nw_backend implemented on raylib 5.
 *
 * This is the whole adapter: ~10 short functions. Writing a backend for any
 * other renderer means writing this file again against a different API, and
 * nothing in the widget library changes.
 */

#ifndef NW_RAYLIB_H
#define NW_RAYLIB_H

#include "native_widgets.h"

/*
 * A backend that draws into whatever raylib target is currently bound. Call
 * between BeginDrawing and EndDrawing.
 *
 * The returned value borrows nothing and can be copied freely; nw_font values
 * you pass to widgets must be `Font *` you own and keep alive.
 */
nw_backend nw_raylib_backend(void);

#endif /* NW_RAYLIB_H */
