# small
"""
================================================================================
HƯỚNG GIẢI 1: QUY HOẠCH ĐỘNG (DYNAMIC PROGRAMMING) CHO BÀI TOÁN
PHÂN BỔ NGÂN SÁCH / DỰ ÁN (KNAPSACK 0/1)
================================================================================
Bài toán:
    Có n dự án, dự án i có chi phí cost[i] và lợi ích benefit[i].
    Chọn tập con các dự án sao cho:
        - Tổng chi phí <= Ngân sách W
        - Tổng lợi ích LỚN NHẤT

Đặc điểm thuật toán:
    - Cho nghiệm TỐI ƯU CHÍNH XÁC (exact optimal solution).
    - Độ phức tạp thời gian: O(n * W)   -> pseudo-polynomial
    - Độ phức tạp bộ nhớ:
        + Bản đầy đủ (có truy vết):  O(n * W)
        + Bản tối ưu bộ nhớ:         O(W)   (nhưng KHÔNG truy vết được trực tiếp)
    - Hạn chế cốt lõi: khi W rất lớn (ví dụ ngân sách hàng triệu/tỷ đơn vị),
      bảng DP sẽ bùng nổ về bộ nhớ và thời gian chạy.

Nội dung file:
    1. knapsack_dp_full()      - DP đầy đủ 2 chiều, có truy vết ra danh sách dự án chọn.
    2. knapsack_dp_optimized() - DP tối ưu bộ nhớ 1 chiều, chỉ trả về giá trị tối ưu.
    3. knapsack_dp_with_trace()- DP 1 chiều nhưng vẫn truy vết được (dùng thêm bảng chọn).
    4. Demo đo thời gian & bộ nhớ khi W tăng dần -> minh họa điểm "bùng nổ".
================================================================================
"""
import time
import tracemalloc
from typing import List, Tuple
# ------------------------------------------------------------------------------
# 1. DP ĐẦY ĐỦ (2 CHIỀU) - DỄ HIỂU, DỄ TRUY VẾT
# ------------------------------------------------------------------------------
def knapsack_dp_full(costs: List[int], benefits: List[int], W: int) -> Tuple[int, List[int]]:
    """
    DP kinh điển dùng bảng 2 chiều dp[i][w].

    dp[i][w] = lợi ích lớn nhất khi chỉ xét i dự án đầu tiên, với ngân sách w.

    Công thức truy hồi:
        dp[i][w] = dp[i-1][w]                                  nếu cost[i] > w
        dp[i][w] = max(dp[i-1][w], dp[i-1][w-cost[i]] + benefit[i])   nếu cost[i] <= w

    Returns:
        best_value   : tổng lợi ích tối ưu
        chosen_items : danh sách chỉ số (0-based) các dự án được chọn
    """
    n = len(costs)
    # dp[i][w]: i chạy 0..n, w chạy 0..W
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        cost_i = costs[i - 1]
        ben_i = benefits[i - 1]
        for w in range(W + 1):
            if cost_i > w:
                dp[i][w] = dp[i - 1][w]
            else:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - cost_i] + ben_i)

    best_value = dp[n][W]

    # ---- Truy vết (backtrack) để tìm ra tập dự án được chọn ----
    chosen_items = []
    w = W
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:   # dự án i đã được chọn
            chosen_items.append(i - 1)  # lưu chỉ số 0-based
            w -= costs[i - 1]
    chosen_items.reverse()

    return best_value, chosen_items
# ------------------------------------------------------------------------------
# 2. DP TỐI ƯU BỘ NHỚ (1 CHIỀU) - CHỈ LẤY GIÁ TRỊ TỐI ƯU
# ------------------------------------------------------------------------------
def knapsack_dp_optimized(costs: List[int], benefits: List[int], W: int) -> int:
    """
    DP dùng mảng 1 chiều (rolling array) để giảm bộ nhớ từ O(n*W) xuống O(W).

    Lưu ý quan trọng: phải duyệt w từ LỚN -> NHỎ để đảm bảo mỗi dự án
    chỉ được dùng tối đa 1 lần (đúng tính chất 0/1 knapsack).
    """
    dp = [0] * (W + 1)

    for cost_i, ben_i in zip(costs, benefits):
        for w in range(W, cost_i - 1, -1):
            dp[w] = max(dp[w], dp[w - cost_i] + ben_i)

    return dp[W]


# ------------------------------------------------------------------------------
# 3. DP 1 CHIỀU NHƯNG VẪN TRUY VẾT ĐƯỢC (dùng bảng "chọn" song song)
# ------------------------------------------------------------------------------
def knapsack_dp_with_trace(costs: List[int], benefits: List[int], W: int) -> Tuple[int, List[int]]:
    """
    Biến thể cân bằng: vẫn dùng mảng 1 chiều cho dp (tiết kiệm bộ nhớ hơn bản full),
    nhưng lưu thêm ma trận 'keep' (n x (W+1)) dạng bit để truy vết được.
    Bộ nhớ: O(n*W) cho keep (dùng kiểu bool/bit để tiết kiệm hơn so với bản full).
    """
    n = len(costs)
    dp = [0] * (W + 1)
    keep = [[False] * (W + 1) for _ in range(n)]

    for i in range(n):
        cost_i, ben_i = costs[i], benefits[i]
        for w in range(W, cost_i - 1, -1):
            candidate = dp[w - cost_i] + ben_i
            if candidate > dp[w]:
                dp[w] = candidate
                keep[i][w] = True

    best_value = dp[W]

    # Truy vết ngược
    chosen_items = []
    w = W
    for i in range(n - 1, -1, -1):
        if keep[i][w]:
            chosen_items.append(i)
            w -= costs[i]
    chosen_items.reverse()

    return best_value, chosen_items
# ------------------------------------------------------------------------------
# 4. DEMO: MINH HỌA "ĐIỂM BÙNG NỔ" KHI W TĂNG DẦN
# ------------------------------------------------------------------------------
def demo_scaling_with_W(costs: List[int], benefits: List[int], W_list: List[int]):
    """
    Cố định n, tăng dần W -> đo thời gian chạy và bộ nhớ đỉnh (peak memory)
    để minh họa thực nghiệm việc DP bị giới hạn bởi kích thước ngân sách W.
    """
    print(f"{'W':>12} | {'Thời gian (s)':>15} | {'Bộ nhớ đỉnh (MB)':>18}")
    print("-" * 55)

    for W in W_list:
        tracemalloc.start()
        t0 = time.perf_counter()

        try:
            value = knapsack_dp_optimized(costs, benefits, W)
            elapsed = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
            peak_mb = peak / (1024 * 1024)
            print(f"{W:>12,} | {elapsed:>15.4f} | {peak_mb:>18.2f}   (best={value})")
        except MemoryError:
            print(f"{W:>12,} |  MemoryError - DP không thể chạy với W này")
        finally:
            tracemalloc.stop()

# ------------------------------------------------------------------------------
# MAIN - VÍ DỤ SỬ DỤNG
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Ví dụ dữ liệu nhỏ, minh họa đúng đắn của thuật toán ---
    project_names = ["Dự án A", "Dự án B", "Dự án C", "Dự án D", "Dự án E"]
    costs = [2, 3, 4, 5, 9]        # chi phí từng dự án
    benefits = [3, 4, 5, 8, 10]    # lợi ích tương ứng
    W = 10                         # ngân sách

    print("=" * 60)
    print("VÍ DỤ MINH HỌA - DP ĐẦY ĐỦ (CÓ TRUY VẾT)")
    print("=" * 60)
    best_value, chosen = knapsack_dp_full(costs, benefits, W)
    print(f"Ngân sách W = {W}")
    print(f"Tổng lợi ích tối ưu = {best_value}")
    print("Danh sách dự án được chọn:")
    total_cost = 0
    for idx in chosen:
        print(f"  - {project_names[idx]}: chi phí={costs[idx]}, lợi ích={benefits[idx]}")
        total_cost += costs[idx]
    print(f"Tổng chi phí sử dụng: {total_cost}/{W}")

    print()
    print("=" * 60)
    print("KIỂM CHỨNG CHÉO VỚI PHIÊN BẢN TỐI ƯU BỘ NHỚ")
    print("=" * 60)
    value_optimized = knapsack_dp_optimized(costs, benefits, W)
    value_trace = knapsack_dp_with_trace(costs, benefits, W)[0]
    print(f"knapsack_dp_full()      -> {best_value}")
    print(f"knapsack_dp_optimized() -> {value_optimized}")
    print(f"knapsack_dp_with_trace()-> {value_trace}")
    assert best_value == value_optimized == value_trace, "Ba phiên bản phải cho cùng kết quả!"
    print("=> Ba phiên bản khớp nhau, thuật toán đúng đắn.")

    print()
    print("=" * 60)
    print("DEMO: ĐO THỰC NGHIỆM KHI W TĂNG DẦN (n cố định)")
    print("=" * 60)
    # Sinh dữ liệu ngẫu nhiên cỡ trung bình để đo hiệu năng
    # (Quy mô được chọn vừa đủ để chạy nhanh; muốn thấy rõ hơn "điểm bùng nổ",
    #  có thể tăng n_items lên 500-1000 và W lên tới 10^7-10^8, nhưng khi đó
    #  nên chạy riêng vì độ phức tạp O(n*W) bằng Python thuần sẽ khá chậm.)
    import random
    random.seed(42)
    n_items = 100
    rand_costs = [random.randint(1, 500) for _ in range(n_items)]
    rand_benefits = [random.randint(1, 500) for _ in range(n_items)]

    W_values_to_test = [1_000, 10_000, 100_000]
    demo_scaling_with_W(rand_costs, rand_benefits, W_values_to_test)

    print()
    print("NHẬN XÉT:")
    print(" - Thời gian và bộ nhớ tăng gần như TUYẾN TÍNH theo W (đúng như O(n*W)).")
    print(" - Khi W đạt tới hàng chục/hàng trăm triệu, bảng DP sẽ vượt quá")
    print("   bộ nhớ RAM thông thường -> đây chính là giới hạn cần Simulated")
    print("   Annealing (Hướng giải 2) để khắc phục.")