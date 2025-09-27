const std = @import("std");
const c = @cImport({
    @cInclude("SDL2/SDL.h");
});

const zig_allocator_visualizer = @import("root.zig");

const builtin = @import("builtin");
const is_macos = builtin.os.tag == .macos;

const GRID_SIZE = 20;
const GRID_WIDTH = 30;
const GRID_HEIGHT = 20;
const WINDOW_WIDTH = GRID_WIDTH * GRID_SIZE;
const WINDOW_HEIGHT = GRID_HEIGHT * GRID_SIZE;

const Direction = enum {
    Up,
    Down,
    Left,
    Right,
};

const Position = struct {
    x: i32,
    y: i32,

    fn equals(self: Position, other: Position) bool {
        return self.x == other.x and self.y == other.y;
    }
};

const SnakeSegment = struct {
    pos: Position,
    // We allocate each segment individually to create memory allocations
    next: ?*SnakeSegment,
};

const Food = struct {
    pos: Position,
    points: u32,
    // Allocate varying amounts of "flavor data" to create different allocation sizes
    flavor_data: []u8,
};

const Particle = struct {
    pos: Position,
    velocity: struct { x: f32, y: f32 },
    life: f32, // Time remaining (0.0 to 1.0)
    size: f32,
    color: struct { r: u8, g: u8, b: u8 },
    // Each particle has its own heap-allocated data
    particle_data: []u8,

    fn update(self: *Particle, dt: f32) bool {
        // Update position
        self.pos.x += @intFromFloat(self.velocity.x * dt);
        self.pos.y += @intFromFloat(self.velocity.y * dt);

        // Update life
        self.life -= dt * 2.0; // Particles live for ~0.5 seconds

        // Apply some physics
        self.velocity.x *= 0.98; // Friction
        self.velocity.y *= 0.98;

        return self.life > 0.0;
    }
};

const GameState = struct {
    snake_head: ?*SnakeSegment,
    snake_length: u32,
    direction: Direction,
    next_direction: Direction,
    food: ?*Food,
    score: u32,
    game_over: bool,
    paused: bool,
    allocator: std.mem.Allocator,
    rng: std.Random.DefaultPrng,
    // Particle system for food eating effects
    particles: std.ArrayList(*Particle),

    fn init(allocator: std.mem.Allocator) !GameState {
        var game = GameState{
            .snake_head = null,
            .snake_length = 0,
            .direction = .Right,
            .next_direction = .Right,
            .food = null,
            .score = 0,
            .game_over = false,
            .paused = false,
            .allocator = allocator,
            .rng = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp())),
            .particles = .{},
        };

        // Initialize snake with 3 segments
        try game.initSnake();
        try game.spawnFood();

        return game;
    }

    fn initSnake(self: *GameState) !void {
        // Create initial snake segments (allocations!)
        const head = try self.allocator.create(SnakeSegment);
        head.* = SnakeSegment{ .pos = .{ .x = 10, .y = 10 }, .next = null };

        const body1 = try self.allocator.create(SnakeSegment);
        body1.* = SnakeSegment{ .pos = .{ .x = 9, .y = 10 }, .next = null };

        const body2 = try self.allocator.create(SnakeSegment);
        body2.* = SnakeSegment{ .pos = .{ .x = 8, .y = 10 }, .next = null };

        head.next = body1;
        body1.next = body2;

        self.snake_head = head;
        self.snake_length = 3;
    }

    fn spawnFood(self: *GameState) !void {
        if (self.food) |food| {
            self.allocator.free(food.flavor_data);
            self.allocator.destroy(food);
        }

        const food = try self.allocator.create(Food);

        // Generate random position
        var valid_pos = false;
        var pos: Position = undefined;
        while (!valid_pos) {
            pos = Position{
                .x = self.rng.random().intRangeAtMost(i32, 0, GRID_WIDTH - 1),
                .y = self.rng.random().intRangeAtMost(i32, 0, GRID_HEIGHT - 1),
            };
            valid_pos = !self.isSnakePosition(pos);
        }

        // Allocate random "flavor data" to create varying allocation sizes
        const flavor_size = self.rng.random().intRangeAtMost(usize, 16, 256);
        const flavor_data = try self.allocator.alloc(u8, flavor_size);

        // Fill with random flavor data
        for (flavor_data, 0..) |*byte, i| {
            byte.* = @intCast((self.rng.random().int(u8) + i) % 256);
        }

        food.* = Food{
            .pos = pos,
            .points = self.rng.random().intRangeAtMost(u32, 10, 50),
            .flavor_data = flavor_data,
        };

        self.food = food;
        std.debug.print("Spawned food at ({},{}) with {} bytes of flavor data\n", .{ pos.x, pos.y, flavor_size });
    }

    fn isSnakePosition(self: *GameState, pos: Position) bool {
        var current = self.snake_head;
        while (current) |segment| {
            if (segment.pos.equals(pos)) return true;
            current = segment.next;
        }
        return false;
    }

    fn update(self: *GameState) !void {
        if (self.game_over or self.paused) return;

        self.direction = self.next_direction;

        // Calculate new head position
        const head = self.snake_head orelse return;
        var new_head_pos = head.pos;

        switch (self.direction) {
            .Up => new_head_pos.y -= 1,
            .Down => new_head_pos.y += 1,
            .Left => new_head_pos.x -= 1,
            .Right => new_head_pos.x += 1,
        }

        // Handle edge wrapping (snake loops around)
        if (new_head_pos.x < 0) {
            new_head_pos.x = GRID_WIDTH - 1;
        } else if (new_head_pos.x >= GRID_WIDTH) {
            new_head_pos.x = 0;
        }

        if (new_head_pos.y < 0) {
            new_head_pos.y = GRID_HEIGHT - 1;
        } else if (new_head_pos.y >= GRID_HEIGHT) {
            new_head_pos.y = 0;
        }

        // Check self collision
        if (self.isSnakePosition(new_head_pos)) {
            self.game_over = true;
            std.debug.print("Game Over! Hit self. Final score: {}\n", .{self.score});
            return;
        }

        // Check food collision
        const ate_food = if (self.food) |food| food.pos.equals(new_head_pos) else false;

        if (ate_food) {
            // Eat food - grow snake (allocate new segment!)
            const new_head = try self.allocator.create(SnakeSegment);
            new_head.* = SnakeSegment{ .pos = new_head_pos, .next = self.snake_head };
            self.snake_head = new_head;
            self.snake_length += 1;

            if (self.food) |food| {
                self.score += food.points;
                std.debug.print("Ate food! Score: {}, grew to {} segments\n", .{ self.score, self.snake_length });

                // Create particle explosion effect!
                try self.createParticleExplosion(food.pos);
            }

            try self.spawnFood();
        } else {
            // Move snake - create new head and remove tail
            const new_head = try self.allocator.create(SnakeSegment);
            new_head.* = SnakeSegment{ .pos = new_head_pos, .next = self.snake_head };

            // Find and remove tail (free memory!)
            var current = new_head;
            var count: u32 = 1;
            while (current.next != null and count < self.snake_length) {
                current = current.next.?;
                count += 1;
            }

            if (current.next) |tail| {
                current.next = null;
                self.allocator.destroy(tail);
            }

            self.snake_head = new_head;
        }

        // Update particles
        try self.updateParticles();
    }

    fn handleInput(self: *GameState, key: c.SDL_Keycode) void {
        switch (key) {
            c.SDLK_UP => {
                if (self.direction != .Down) self.next_direction = .Up;
            },
            c.SDLK_DOWN => {
                if (self.direction != .Up) self.next_direction = .Down;
            },
            c.SDLK_LEFT => {
                if (self.direction != .Right) self.next_direction = .Left;
            },
            c.SDLK_RIGHT => {
                if (self.direction != .Left) self.next_direction = .Right;
            },
            c.SDLK_SPACE => {
                self.paused = !self.paused;
            },
            c.SDLK_r => {
                if (self.game_over) self.restart() catch {};
            },
            else => {},
        }
    }

    fn restart(self: *GameState) !void {
        // Free all snake segments
        self.freeSnake();

        // Free food
        if (self.food) |food| {
            self.allocator.free(food.flavor_data);
            self.allocator.destroy(food);
            self.food = null;
        }

        // Clean up all particles
        for (self.particles.items) |particle| {
            self.allocator.free(particle.particle_data);
            self.allocator.destroy(particle);
        }
        self.particles.clearRetainingCapacity();

        // Reset state
        self.snake_head = null;
        self.snake_length = 0;
        self.direction = .Right;
        self.next_direction = .Right;
        self.score = 0;
        self.game_over = false;
        self.paused = false;

        // Reinitialize
        try self.initSnake();
        try self.spawnFood();

        std.debug.print("Game restarted!\n", .{});
    }

    fn freeSnake(self: *GameState) void {
        var current = self.snake_head;
        while (current) |segment| {
            const next = segment.next;
            self.allocator.destroy(segment);
            current = next;
        }
        self.snake_head = null;
    }

    fn createParticleExplosion(self: *GameState, pos: Position) !void {
        // Create 8-12 particles with random velocities and heap-allocated data
        const particle_count = self.rng.random().intRangeAtMost(u32, 8, 12);

        for (0..particle_count) |_| {
            const particle = try self.allocator.create(Particle);

            // Random velocity for explosion effect
            const angle = self.rng.random().float(f32) * std.math.pi * 2.0;
            const speed = 50.0 + self.rng.random().float(f32) * 100.0;

            // Allocate random data for each particle (this creates heap allocations!)
            const data_size = self.rng.random().intRangeAtMost(usize, 8, 64);
            const particle_data = try self.allocator.alloc(u8, data_size);

            // Fill with random data
            for (particle_data, 0..) |*byte, i| {
                byte.* = @intCast((self.rng.random().int(u8) + i) % 256);
            }

            particle.* = Particle{
                .pos = pos,
                .velocity = .{
                    .x = @cos(angle) * speed,
                    .y = @sin(angle) * speed,
                },
                .life = 1.0,
                .size = 2.0 + self.rng.random().float(f32) * 4.0,
                .color = .{
                    .r = self.rng.random().intRangeAtMost(u8, 200, 255),
                    .g = self.rng.random().intRangeAtMost(u8, 100, 200),
                    .b = self.rng.random().intRangeAtMost(u8, 50, 150),
                },
                .particle_data = particle_data,
            };

            try self.particles.append(self.allocator, particle);
            std.debug.print("Created particle with {} bytes at ({}, {})\n", .{ data_size, pos.x, pos.y });
        }
    }

    fn updateParticles(self: *GameState) !void {
        const dt: f32 = 0.016; // ~60fps

        var i: usize = 0;
        while (i < self.particles.items.len) {
            const particle = self.particles.items[i];

            if (particle.update(dt)) {
                // Particle is still alive
                i += 1;
            } else {
                // Particle died, clean it up
                const dead_particle = self.particles.orderedRemove(i);
                const data_len = dead_particle.particle_data.len;
                self.allocator.free(dead_particle.particle_data);
                self.allocator.destroy(dead_particle);
                std.debug.print("Cleaned up particle with {} bytes of data\n", .{data_len});
                // Don't increment i since we removed an item
            }
        }
    }

    fn deinit(self: *GameState) void {
        self.freeSnake();
        if (self.food) |food| {
            self.allocator.free(food.flavor_data);
            self.allocator.destroy(food);
        }
        // Clean up all particles
        for (self.particles.items) |particle| {
            self.allocator.free(particle.particle_data);
            self.allocator.destroy(particle);
        }
        self.particles.deinit(self.allocator);
    }

    fn render(self: *GameState, renderer: *c.SDL_Renderer) void {
        // Clear screen
        _ = c.SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        _ = c.SDL_RenderClear(renderer);

        // Draw grid
        _ = c.SDL_SetRenderDrawColor(renderer, 30, 30, 30, 255);
        for (0..GRID_WIDTH + 1) |i| {
            const x = @as(i32, @intCast(i * GRID_SIZE));
            _ = c.SDL_RenderDrawLine(renderer, x, 0, x, WINDOW_HEIGHT);
        }
        for (0..GRID_HEIGHT + 1) |i| {
            const y = @as(i32, @intCast(i * GRID_SIZE));
            _ = c.SDL_RenderDrawLine(renderer, 0, y, WINDOW_WIDTH, y);
        }

        // Draw snake
        var current = self.snake_head;
        var is_head = true;
        while (current) |segment| {
            if (is_head) {
                _ = c.SDL_SetRenderDrawColor(renderer, 100, 255, 100, 255); // Bright green head
            } else {
                _ = c.SDL_SetRenderDrawColor(renderer, 50, 200, 50, 255); // Green body
            }

            const rect = c.SDL_Rect{
                .x = segment.pos.x * GRID_SIZE + 1,
                .y = segment.pos.y * GRID_SIZE + 1,
                .w = GRID_SIZE - 2,
                .h = GRID_SIZE - 2,
            };
            _ = c.SDL_RenderFillRect(renderer, &rect);

            current = segment.next;
            is_head = false;
        }

        // Draw food
        if (self.food) |food| {
            _ = c.SDL_SetRenderDrawColor(renderer, 255, 100, 100, 255); // Red food
            const rect = c.SDL_Rect{
                .x = food.pos.x * GRID_SIZE + 2,
                .y = food.pos.y * GRID_SIZE + 2,
                .w = GRID_SIZE - 4,
                .h = GRID_SIZE - 4,
            };
            _ = c.SDL_RenderFillRect(renderer, &rect);
        }

        // Draw particles
        for (self.particles.items) |particle| {
            _ = c.SDL_SetRenderDrawColor(renderer, particle.color.r, particle.color.g, particle.color.b, @intFromFloat(particle.life * 255.0));

            const screen_x = @as(i32, @intFromFloat(@as(f32, @floatFromInt(particle.pos.x * GRID_SIZE)) + particle.size));
            const screen_y = @as(i32, @intFromFloat(@as(f32, @floatFromInt(particle.pos.y * GRID_SIZE)) + particle.size));
            const size = @as(i32, @intFromFloat(particle.size));

            const rect = c.SDL_Rect{
                .x = screen_x - @divTrunc(size, 2),
                .y = screen_y - @divTrunc(size, 2),
                .w = size,
                .h = size,
            };
            _ = c.SDL_RenderFillRect(renderer, &rect);
        }

        // Draw UI text areas (placeholder rectangles)
        if (self.paused) {
            _ = c.SDL_SetRenderDrawColor(renderer, 100, 100, 100, 200);
            const pause_rect = c.SDL_Rect{ .x = WINDOW_WIDTH / 4, .y = WINDOW_HEIGHT / 2 - 20, .w = WINDOW_WIDTH / 2, .h = 40 };
            _ = c.SDL_RenderFillRect(renderer, &pause_rect);
        }

        if (self.game_over) {
            _ = c.SDL_SetRenderDrawColor(renderer, 200, 50, 50, 200);
            const game_over_rect = c.SDL_Rect{ .x = WINDOW_WIDTH / 4, .y = WINDOW_HEIGHT / 2 - 30, .w = WINDOW_WIDTH / 2, .h = 60 };
            _ = c.SDL_RenderFillRect(renderer, &game_over_rect);
        }

        c.SDL_RenderPresent(renderer);
    }
};

// Shared state for dual windows
const DualWindowState = struct {
    tracking: *zig_allocator_visualizer.TrackingAllocator,
    should_quit: std.atomic.Value(bool),
    game_state: *GameState,

    fn init(tracking: *zig_allocator_visualizer.TrackingAllocator, game_state: *GameState) DualWindowState {
        return .{
            .tracking = tracking,
            .should_quit = std.atomic.Value(bool).init(false),
            .game_state = game_state,
        };
    }
};

pub fn main() !void {
    const base_allocator = std.heap.page_allocator;

    // Create tracking allocator
    var tracking = zig_allocator_visualizer.TrackingAllocator.init(base_allocator);
    defer tracking.deinit();

    const monitored_allocator = tracking.allocator();

    // Create game state
    var game_state = try GameState.init(monitored_allocator);
    defer game_state.deinit();

    std.debug.print("=== Snake Game with Live Memory Visualization ===\n", .{});
    std.debug.print("Arrow keys: Move snake\n", .{});
    std.debug.print("SPACE: Pause/Resume\n", .{});
    std.debug.print("R: Restart (when game over)\n", .{});
    std.debug.print("ESC: Quit\n", .{});

    var shared_state = DualWindowState.init(&tracking, &game_state);

    if (is_macos) {
        // On macOS, run both windows on main thread
        try gameLoopWithVisualization(&shared_state);
    } else {
        // On other platforms, run visualization in separate thread
        var viz_thread = try std.Thread.spawn(.{}, visualizationThread, .{&shared_state});
        defer viz_thread.join();

        try gameLoop(&shared_state);
    }
}

fn gameLoopWithVisualization(shared_state: *DualWindowState) !void {
    // Initialize SDL for both windows on main thread (macOS requirement)
    if (c.SDL_Init(c.SDL_INIT_VIDEO) != 0) {
        std.debug.print("SDL init failed: {s}\n", .{c.SDL_GetError()});
        return;
    }
    defer c.SDL_Quit();

    // Create game window
    const game_window = c.SDL_CreateWindow(
        "Snake Game",
        100, // X position
        100, // Y position
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        c.SDL_WINDOW_SHOWN,
    ) orelse {
        std.debug.print("Failed to create game window: {s}\n", .{c.SDL_GetError()});
        return;
    };
    defer c.SDL_DestroyWindow(game_window);

    const game_renderer = c.SDL_CreateRenderer(
        game_window,
        -1,
        c.SDL_RENDERER_ACCELERATED | c.SDL_RENDERER_PRESENTVSYNC,
    ) orelse {
        std.debug.print("Failed to create game renderer: {s}\n", .{c.SDL_GetError()});
        return;
    };
    defer c.SDL_DestroyRenderer(game_renderer);

    // Create visualization window
    const viz_window = c.SDL_CreateWindow(
        "Memory Visualization",
        700, // X position (to the right of game window)
        100, // Y position
        800,
        600,
        c.SDL_WINDOW_SHOWN | c.SDL_WINDOW_RESIZABLE,
    ) orelse {
        std.debug.print("Failed to create visualization window: {s}\n", .{c.SDL_GetError()});
        return;
    };
    defer c.SDL_DestroyWindow(viz_window);

    const viz_renderer = c.SDL_CreateRenderer(
        viz_window,
        -1,
        c.SDL_RENDERER_ACCELERATED | c.SDL_RENDERER_PRESENTVSYNC,
    ) orelse {
        std.debug.print("Failed to create visualization renderer: {s}\n", .{c.SDL_GetError()});
        return;
    };
    defer c.SDL_DestroyRenderer(viz_renderer);

    var event: c.SDL_Event = undefined;
    var last_update = std.time.milliTimestamp();
    const update_interval: i64 = 150; // 150ms between moves
    var sort_by_address = false;

    while (!shared_state.should_quit.load(.acquire)) {
        // Handle events for both windows
        while (c.SDL_PollEvent(&event) != 0) {
            switch (event.type) {
                c.SDL_QUIT => shared_state.should_quit.store(true, .release),
                c.SDL_KEYDOWN => {
                    switch (event.key.keysym.sym) {
                        c.SDLK_ESCAPE => shared_state.should_quit.store(true, .release),
                        c.SDLK_SPACE => {
                            // Check which window has focus to decide action
                            const focused_window = c.SDL_GetKeyboardFocus();
                            if (focused_window == game_window) {
                                shared_state.game_state.handleInput(event.key.keysym.sym);
                            } else if (focused_window == viz_window) {
                                sort_by_address = !sort_by_address;
                            } else {
                                // Default to game input
                                shared_state.game_state.handleInput(event.key.keysym.sym);
                            }
                        },
                        else => shared_state.game_state.handleInput(event.key.keysym.sym),
                    }
                },
                else => {},
            }
        }

        // Update game
        const current_time = std.time.milliTimestamp();
        if (current_time - last_update >= update_interval) {
            try shared_state.game_state.update();
            last_update = current_time;
        }

        // Render game window
        shared_state.game_state.render(game_renderer);

        // Render visualization window
        renderMemoryVisualization(viz_renderer, viz_window, shared_state.tracking, sort_by_address) catch |err| {
            std.debug.print("Visualization render error: {}\n", .{err});
        };

        std.Thread.sleep(16_666_667); // ~60 FPS
    }
}

fn gameLoop(shared_state: *DualWindowState) !void {
    // Initialize SDL for game window
    if (c.SDL_Init(c.SDL_INIT_VIDEO) != 0) {
        std.debug.print("SDL init failed: {s}\n", .{c.SDL_GetError()});
        return;
    }
    defer c.SDL_Quit();

    const game_window = c.SDL_CreateWindow(
        "Snake Game",
        100, // X position
        100, // Y position
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        c.SDL_WINDOW_SHOWN,
    ) orelse {
        std.debug.print("Failed to create game window: {s}\n", .{c.SDL_GetError()});
        return;
    };
    defer c.SDL_DestroyWindow(game_window);

    const game_renderer = c.SDL_CreateRenderer(
        game_window,
        -1,
        c.SDL_RENDERER_ACCELERATED | c.SDL_RENDERER_PRESENTVSYNC,
    ) orelse {
        std.debug.print("Failed to create game renderer: {s}\n", .{c.SDL_GetError()});
        return;
    };
    defer c.SDL_DestroyRenderer(game_renderer);

    var event: c.SDL_Event = undefined;
    var last_update = std.time.milliTimestamp();
    const update_interval: i64 = 150; // 150ms between moves

    while (!shared_state.should_quit.load(.acquire)) {
        // Handle events
        while (c.SDL_PollEvent(&event) != 0) {
            switch (event.type) {
                c.SDL_QUIT => shared_state.should_quit.store(true, .release),
                c.SDL_KEYDOWN => {
                    switch (event.key.keysym.sym) {
                        c.SDLK_ESCAPE => shared_state.should_quit.store(true, .release),
                        else => shared_state.game_state.handleInput(event.key.keysym.sym),
                    }
                },
                else => {},
            }
        }

        // Update game
        const current_time = std.time.milliTimestamp();
        if (current_time - last_update >= update_interval) {
            try shared_state.game_state.update();
            last_update = current_time;
        }

        // Render game
        shared_state.game_state.render(game_renderer);

        std.Thread.sleep(16_666_667); // ~60 FPS
    }
}

fn visualizationThread(shared_state: *DualWindowState) !void {
    // Small delay to let game window initialize first
    std.Thread.sleep(500_000_000); // 500ms

    // Initialize visualization window
    const viz_window = c.SDL_CreateWindow(
        "Memory Visualization",
        700, // X position (to the right of game window)
        100, // Y position
        800,
        600,
        c.SDL_WINDOW_SHOWN | c.SDL_WINDOW_RESIZABLE,
    ) orelse {
        std.debug.print("Failed to create visualization window: {s}\n", .{c.SDL_GetError()});
        return;
    };
    defer c.SDL_DestroyWindow(viz_window);

    const viz_renderer = c.SDL_CreateRenderer(
        viz_window,
        -1,
        c.SDL_RENDERER_ACCELERATED | c.SDL_RENDERER_PRESENTVSYNC,
    ) orelse {
        std.debug.print("Failed to create visualization renderer: {s}\n", .{c.SDL_GetError()});
        return;
    };
    defer c.SDL_DestroyRenderer(viz_renderer);

    var sort_by_address = false;

    while (!shared_state.should_quit.load(.acquire)) {
        // Handle visualization window events
        var event: c.SDL_Event = undefined;
        while (c.SDL_PollEvent(&event) != 0) {
            switch (event.type) {
                c.SDL_QUIT => shared_state.should_quit.store(true, .release),
                c.SDL_KEYDOWN => {
                    switch (event.key.keysym.sym) {
                        c.SDLK_SPACE => sort_by_address = !sort_by_address,
                        c.SDLK_ESCAPE => shared_state.should_quit.store(true, .release),
                        else => {},
                    }
                },
                else => {},
            }
        }

        // Render visualization
        try renderMemoryVisualization(viz_renderer, viz_window, shared_state.tracking, sort_by_address);

        std.Thread.sleep(33_333_333); // ~30 FPS for visualization
    }
}

fn renderMemoryVisualization(renderer: *c.SDL_Renderer, window: *c.SDL_Window, tracking: *zig_allocator_visualizer.TrackingAllocator, sort_by_address: bool) !void {
    // Get window size
    var window_width: i32 = undefined;
    var window_height: i32 = undefined;
    c.SDL_GetWindowSize(window, &window_width, &window_height);

    // Clear screen
    _ = c.SDL_SetRenderDrawColor(renderer, 20, 20, 20, 255);
    _ = c.SDL_RenderClear(renderer);

    const tracker = tracking.getTracker();

    // Draw title area
    _ = c.SDL_SetRenderDrawColor(renderer, 60, 60, 60, 255);
    const title_rect = c.SDL_Rect{ .x = 10, .y = 10, .w = window_width - 20, .h = 30 };
    _ = c.SDL_RenderFillRect(renderer, &title_rect);

    // Draw stats
    renderStats(renderer, tracker, window_width);

    // Draw memory visualization
    try renderAllocations(renderer, tracker, window_width, window_height, sort_by_address);

    // Draw mode indicator
    _ = c.SDL_SetRenderDrawColor(renderer, 100, 100, 100, 255);
    const mode_rect = c.SDL_Rect{ .x = 10, .y = window_height - 30, .w = 200, .h = 20 };
    _ = c.SDL_RenderFillRect(renderer, &mode_rect);

    c.SDL_RenderPresent(renderer);
}

fn renderStats(renderer: *c.SDL_Renderer, tracker: *zig_allocator_visualizer.AllocationTracker, window_width: i32) void {
    const stats_y = 50;
    const bar_height = 15;
    const bar_width = window_width - 20;

    const current_usage = tracker.total_bytes_allocated - tracker.total_bytes_freed;
    const total_allocated = tracker.total_bytes_allocated;

    // Memory usage bar
    if (total_allocated > 0) {
        const usage_percent = @as(f32, @floatFromInt(current_usage)) / @as(f32, @floatFromInt(total_allocated));
        const filled_width = @as(i32, @intFromFloat(@as(f32, @floatFromInt(bar_width)) * usage_percent));

        // Background
        _ = c.SDL_SetRenderDrawColor(renderer, 50, 50, 50, 255);
        const bg_rect = c.SDL_Rect{ .x = 10, .y = stats_y, .w = bar_width, .h = bar_height };
        _ = c.SDL_RenderFillRect(renderer, &bg_rect);

        // Filled portion
        _ = c.SDL_SetRenderDrawColor(renderer, 100, 200, 100, 255);
        const fill_rect = c.SDL_Rect{ .x = 10, .y = stats_y, .w = filled_width, .h = bar_height };
        _ = c.SDL_RenderFillRect(renderer, &fill_rect);
    }

    // Allocation count bar
    const count_y = stats_y + 20;
    const count_width = @min(tracker.allocation_count * 8, @as(usize, @intCast(bar_width)));
    _ = c.SDL_SetRenderDrawColor(renderer, 100, 150, 255, 255);
    const count_rect = c.SDL_Rect{
        .x = 10,
        .y = count_y,
        .w = @intCast(count_width),
        .h = bar_height,
    };
    _ = c.SDL_RenderFillRect(renderer, &count_rect);
}

fn renderAllocations(renderer: *c.SDL_Renderer, tracker: *zig_allocator_visualizer.AllocationTracker, window_width: i32, window_height: i32, sort_by_address: bool) !void {
    const map_x = 10;
    const map_y = 100;
    const map_width = window_width - 20;
    const map_height = window_height - 150;

    // Draw border
    _ = c.SDL_SetRenderDrawColor(renderer, 100, 100, 100, 255);
    const border_rect = c.SDL_Rect{ .x = map_x, .y = map_y, .w = map_width, .h = map_height };
    _ = c.SDL_RenderDrawRect(renderer, &border_rect);

    if (tracker.memory_ordered.items.len == 0) return;

    // Sort allocations
    const allocator = std.heap.page_allocator;
    var sorted_allocs: std.ArrayList(zig_allocator_visualizer.AllocationInfo) = .{};
    defer sorted_allocs.deinit(allocator);

    for (tracker.memory_ordered.items) |info| {
        try sorted_allocs.append(allocator, info);
    }

    if (sort_by_address) {
        std.sort.insertion(zig_allocator_visualizer.AllocationInfo, sorted_allocs.items, {}, compareByAddress);
    } else {
        std.sort.insertion(zig_allocator_visualizer.AllocationInfo, sorted_allocs.items, {}, compareByTime);
    }

    // Render as grid
    const item_width = 12;
    const item_height = 20;
    const spacing = 2;
    const items_per_row = @as(usize, @intCast(@divTrunc(map_width - 10, item_width + spacing)));

    for (sorted_allocs.items, 0..) |info, i| {
        const row = @divTrunc(i, items_per_row);
        const col = i % items_per_row;

        const x = map_x + 5 + @as(i32, @intCast(col)) * (item_width + spacing);
        const y = map_y + 5 + @as(i32, @intCast(row)) * (item_height + spacing);

        if (y + item_height > map_y + map_height - 5) break;

        // Color based on size and type
        const color = getColorForAllocation(info);
        _ = c.SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, 200);

        const rect = c.SDL_Rect{ .x = x, .y = y, .w = item_width, .h = item_height };
        _ = c.SDL_RenderFillRect(renderer, &rect);

        // Border
        _ = c.SDL_SetRenderDrawColor(renderer, 255, 255, 255, 100);
        _ = c.SDL_RenderDrawRect(renderer, &rect);
    }
}

fn compareByAddress(context: void, a: zig_allocator_visualizer.AllocationInfo, b: zig_allocator_visualizer.AllocationInfo) bool {
    _ = context;
    return @intFromPtr(a.ptr) < @intFromPtr(b.ptr);
}

fn compareByTime(context: void, a: zig_allocator_visualizer.AllocationInfo, b: zig_allocator_visualizer.AllocationInfo) bool {
    _ = context;
    return a.timestamp < b.timestamp;
}

const Color = struct { r: u8, g: u8, b: u8 };

fn getColorForAllocation(info: zig_allocator_visualizer.AllocationInfo) Color {
    // Different colors for different allocation types
    if (info.size <= 64) {
        return .{ .r = 100, .g = 255, .b = 100 }; // Green for snake segments
    } else if (info.size <= 256) {
        return .{ .r = 255, .g = 200, .b = 100 }; // Orange for food flavor data
    } else if (info.size <= 1024) {
        return .{ .r = 100, .g = 200, .b = 255 }; // Blue for larger structures
    } else {
        return .{ .r = 255, .g = 100, .b = 100 }; // Red for very large allocations
    }
}