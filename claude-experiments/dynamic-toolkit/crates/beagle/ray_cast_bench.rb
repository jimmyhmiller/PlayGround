# Ruby port of ray_cast_bench.bg — same algorithm, same constants, same
# control flow, so the checksum should match the beagle versions.

SHADOW_FAR = 5000.0
PI = 3.14159265358979

Vec2 = Struct.new(:x, :y)
Segment = Struct.new(:a, :b)

def make_v(x, y) = Vec2.new(x, y)
def make_seg(a, b) = Segment.new(a, b)

def abs_f(x) = x < 0.0 ? -x : x

def ray_segment_hit(origin, dir, a, b)
  sx = b.x - a.x
  sy = b.y - a.y
  denom = dir.x * sy - dir.y * sx
  if abs_f(denom) < 0.000001
    -1.0
  else
    diffx = a.x - origin.x
    diffy = a.y - origin.y
    t = (diffx * sy - diffy * sx) / denom
    u = (diffx * dir.y - diffy * dir.x) / denom
    (t >= 0.0 && u >= 0.0 && u <= 1.0) ? t : -1.0
  end
end

def cast_ray(origin, dir, segments)
  best = SHADOW_FAR
  i = 0
  n = segments.length
  while i < n
    s = segments[i]
    t = ray_segment_hit(origin, dir, s.a, s.b)
    best = t if t > 0.0 && t < best
    i += 1
  end
  best
end

def build_segments
  walls = [
    [100.0, 100.0, 60.0, 40.0],
    [-150.0, 80.0, 50.0, 50.0],
    [200.0, -120.0, 40.0, 60.0],
    [-80.0, -200.0, 70.0, 30.0],
    [300.0, 50.0, 50.0, 100.0],
    [-250.0, -50.0, 80.0, 40.0],
    [50.0, 250.0, 100.0, 30.0],
    [0.0, -300.0, 90.0, 50.0]
  ]
  segs = []
  walls.each do |w|
    cx, cy, hx, hy = w
    tl = make_v(cx - hx, cy - hy)
    tr = make_v(cx + hx, cy - hy)
    br = make_v(cx + hx, cy + hy)
    bl = make_v(cx - hx, cy + hy)
    segs << make_seg(tl, tr)
    segs << make_seg(tr, br)
    segs << make_seg(br, bl)
    segs << make_seg(bl, tl)
  end
  segs
end

def build_angles(n)
  two_pi = 2.0 * PI
  angles = []
  i = 0
  while i < n
    angles << (-PI + i.to_f / n.to_f * two_pi)
    i += 1
  end
  angles
end

def do_pass(origin, angles, segments)
  sum = 0.0
  i = 0
  n = angles.length
  while i < n
    a = angles[i]
    dir = make_v(Math.cos(a), Math.sin(a))
    sum += cast_ray(origin, dir, segments)
    i += 1
  end
  sum
end

def run_bench(frames, origin, angles, segments)
  total = 0.0
  frame = 0
  while frame < frames
    total += do_pass(origin, angles, segments)
    frame += 1
  end
  total
end

segments = build_segments
angles = build_angles(64)
origin = make_v(0.0, 0.0)

n_segs = segments.length
n_angles = angles.length

warm = run_bench(200, origin, angles, segments)
puts "warm checksum: #{warm}"

frames = 10000
start = Process.clock_gettime(Process::CLOCK_MONOTONIC, :nanosecond)
total = run_bench(frames, origin, angles, segments)
finish = Process.clock_gettime(Process::CLOCK_MONOTONIC, :nanosecond)

elapsed_ms = (finish - start) / 1_000_000.0
calls = frames * n_angles * n_segs

puts "frames: #{frames}"
puts "segments/frame: #{n_segs}"
puts "rays/frame: #{n_angles}"
puts "ray_segment_hit calls: #{calls}"
puts "checksum: #{total}"
puts "elapsed_ms: #{elapsed_ms}"
