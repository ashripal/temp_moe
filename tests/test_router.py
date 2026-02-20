from implementation.router import SimpleTelemetryRouter

def test_router_picks_comm_when_mpi_wait_high():
    r = SimpleTelemetryRouter()
    d = r.route({"mpi_wait_pct": 40.0, "omp_barrier_pct": 5.0, "omp_imbalance_ratio": 1.1, "memory_bound_score": 0.1})
    assert "Communication & Resilience Expert" in d.selected_experts

def test_router_picks_parallelism_when_barrier_high():
    r = SimpleTelemetryRouter()
    d = r.route({"mpi_wait_pct": 5.0, "omp_barrier_pct": 20.0, "omp_imbalance_ratio": 1.2, "memory_bound_score": 0.1})
    assert "Parallelism & Job Expert" in d.selected_experts

def test_router_picks_kernel_when_memory_bound():
    r = SimpleTelemetryRouter()
    d = r.route({"mpi_wait_pct": 5.0, "omp_barrier_pct": 5.0, "omp_imbalance_ratio": 1.1, "memory_bound_score": 0.8})
    assert "Kernel & System Efficiency Expert" in d.selected_experts