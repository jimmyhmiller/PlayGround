// N-body, 5 Jovian bodies — mirrors benchmarks/ail/nbody.ail (same column
// layout and op order, so the energy checksum matches bit-for-bit).
package main

import (
	"fmt"
	"math"
	"time"
)

const (
	solarMass   = 39.478417604357434
	daysPerYear = 365.24
	nBodies     = 5
)

type sys struct {
	x, y, z, vx, vy, vz, m [nBodies]float64
}

func initSys() *sys {
	bodies := [nBodies][7]float64{
		// Sun.
		{0, 0, 0, 0, 0, 0, 1.0},
		// Jupiter.
		{4.84143144246472090, -1.16032004402742839, -0.103622044471123109,
			0.00166007664274403694, 0.00769901118419740425, -0.0000690460016972063023,
			0.000954791938424326609},
		// Saturn.
		{8.34336671824457987, 4.12479856412430479, -0.403523417114321381,
			-0.00276742510726862411, 0.00499852801234917238, 0.0000230417297573763929,
			0.000285885980666130812},
		// Uranus.
		{12.8943695621391310, -15.1111514016986312, -0.223307578892655734,
			0.00296460137564761618, 0.00237847173959480950, -0.0000296589568540237556,
			0.0000436624404335156298},
		// Neptune.
		{15.3796971148509165, -25.9193146099879641, 0.179258772950371181,
			0.00268067772490389322, 0.00162824170038242295, -0.0000951592254519715870,
			0.0000515138902046611451},
	}
	s := &sys{}
	for i, b := range bodies {
		s.x[i] = b[0]
		s.y[i] = b[1]
		s.z[i] = b[2]
		s.vx[i] = b[3] * daysPerYear
		s.vy[i] = b[4] * daysPerYear
		s.vz[i] = b[5] * daysPerYear
		s.m[i] = b[6] * solarMass
	}
	return s
}

func offsetMomentum(s *sys) {
	px, py, pz := 0.0, 0.0, 0.0
	for i := 0; i < nBodies; i++ {
		px += s.vx[i] * s.m[i]
		py += s.vy[i] * s.m[i]
		pz += s.vz[i] * s.m[i]
	}
	s.vx[0] = -px / solarMass
	s.vy[0] = -py / solarMass
	s.vz[0] = -pz / solarMass
}

func advance(s *sys, dt float64) {
	for i := 0; i < nBodies; i++ {
		for j := i + 1; j < nBodies; j++ {
			dx := s.x[i] - s.x[j]
			dy := s.y[i] - s.y[j]
			dz := s.z[i] - s.z[j]
			d2 := dx*dx + dy*dy + dz*dz
			mag := dt / (d2 * math.Sqrt(d2))
			mi, mj := s.m[i], s.m[j]
			s.vx[i] -= dx * mj * mag
			s.vy[i] -= dy * mj * mag
			s.vz[i] -= dz * mj * mag
			s.vx[j] += dx * mi * mag
			s.vy[j] += dy * mi * mag
			s.vz[j] += dz * mi * mag
		}
	}
	for i := 0; i < nBodies; i++ {
		s.x[i] += dt * s.vx[i]
		s.y[i] += dt * s.vy[i]
		s.z[i] += dt * s.vz[i]
	}
}

func energy(s *sys) float64 {
	e := 0.0
	for i := 0; i < nBodies; i++ {
		e += 0.5 * s.m[i] * (s.vx[i]*s.vx[i] + s.vy[i]*s.vy[i] + s.vz[i]*s.vz[i])
		for j := i + 1; j < nBodies; j++ {
			dx := s.x[i] - s.x[j]
			dy := s.y[i] - s.y[j]
			dz := s.z[i] - s.z[j]
			e -= s.m[i] * s.m[j] / math.Sqrt(dx*dx+dy*dy+dz*dz)
		}
	}
	return e
}

func main() {
	n := 500_000
	s := initSys()
	offsetMomentum(s)
	t0 := time.Now()
	for k := 0; k < n; k++ {
		advance(s, 0.01)
	}
	ms := time.Since(t0).Milliseconds()
	chk := int64(energy(s) * 1_000_000_000.0)
	fmt.Printf("RESULT nbody %d ms checksum=%d\n", ms, chk)
}
