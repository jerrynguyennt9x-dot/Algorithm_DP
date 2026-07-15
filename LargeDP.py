#LargeDP
"""
================================================================================
HƯỚNG GIẢI 1: QUY HOẠCH ĐỘNG (DP) - GIẢI BÀI TOÁN KNAPSACK 0/1
Áp dụng trên bộ dữ liệu thực: dt02_knapsack_small.csv
================================================================================

Định dạng file CSV đầu vào:
    item_id,weight,value
    1,4,1
    2,9,8
    ...
    (dòng cuối, có thể có comment dạng "# W=50" để chỉ định ngân sách W)

Nếu file không có dòng "# W=<số>", chương trình sẽ yêu cầu truyền W thủ công.

Nội dung:
    1. load_knapsack_csv()      - đọc dữ liệu + ngân sách W từ file CSV
    2. knapsack_dp_full()       - DP 2 chiều, có truy vết ra danh sách dự án chọn
    3. print_solution()         - in kết quả rõ ràng, dễ đưa vào báo cáo
    4. main                     - chạy toàn bộ pipeline trên dữ liệu thực tế
================================================================================
"""

import csv
import re
import sys
import time
from typing import List, Tuple, Optional


# ------------------------------------------------------------------------------
# 1. ĐỌC DỮ LIỆU TỪ FILE CSV
# ------------------------------------------------------------------------------
def load_knapsack_csv(filepath: str) -> Tuple[List[int], List[int], List[int], int]:
    """
    Đọc file CSV có cột: item_id, weight, value
    Đồng thời tìm dòng comment dạng "# W=<số>" để lấy ngân sách W.

    Returns:
        item_ids : danh sách id dự án (giữ nguyên từ file)
        weights  : danh sách chi phí (weight) của từng dự án
        values   : danh sách lợi ích (value) của từng dự án
        W        : ngân sách tổng
    """
    item_ids, weights, values = [], [], []
    W: Optional[int] = None

    with open(filepath, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    data_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Tìm dòng khai báo ngân sách dạng "# W=50"
        match = re.search(r"#\s*W\s*=\s*(\d+)", stripped, re.IGNORECASE)
        if match:
            W = int(match.group(1))
            continue
        if stripped.startswith("#"):
            continue
        data_lines.append(stripped)

    reader = csv.DictReader(data_lines)
    for row in reader:
        item_ids.append(int(row["item_id"]))
        weights.append(int(row["weight"]))
        values.append(int(row["value"]))

    if W is None:
        raise ValueError(
            "Không tìm thấy ngân sách W trong file (dòng dạng '# W=<số>'). "
            "Vui lòng truyền W thủ công khi gọi hàm."
        )

    return item_ids, weights, values, W


# ------------------------------------------------------------------------------
# 2. DP 0/1 KNAPSACK - BẢNG 2 CHIỀU, CÓ TRUY VẾT
# ------------------------------------------------------------------------------
def knapsack_dp_full(weights: List[int], values: List[int], W: int) -> Tuple[int, List[int]]:
    """
    dp[i][w] = lợi ích lớn nhất khi chỉ xét i dự án đầu tiên, ngân sách w.

    Công thức truy hồi:
        dp[i][w] = dp[i-1][w]                                     nếu weight[i] > w
        dp[i][w] = max(dp[i-1][w], dp[i-1][w-weight[i]] + value[i])   nếu weight[i] <= w

    Độ phức tạp: O(n * W) thời gian, O(n * W) bộ nhớ.

    Returns:
        best_value    : tổng lợi ích tối ưu
        chosen_indices: danh sách chỉ số (0-based, theo thứ tự trong list đầu vào)
                        của các dự án được chọn
    """
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        w_i, v_i = weights[i - 1], values[i - 1]
        for w in range(W + 1):
            if w_i > w:
                dp[i][w] = dp[i - 1][w]
            else:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - w_i] + v_i)

    best_value = dp[n][W]

    # Truy vết
    chosen_indices = []
    w = W
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            chosen_indices.append(i - 1)
            w -= weights[i - 1]
    chosen_indices.reverse()

    return best_value, chosen_indices


# ------------------------------------------------------------------------------
# 3. IN KẾT QUẢ
# ------------------------------------------------------------------------------
def print_solution(item_ids, weights, values, W, best_value, chosen_indices):
    print("=" * 65)
    print("KẾT QUẢ GIẢI BÀI TOÁN KNAPSACK 0/1 BẰNG QUY HOẠCH ĐỘNG (DP)")
    print("=" * 65)
    print(f"Số lượng dự án (n) : {len(weights)}")
    print(f"Ngân sách (W)      : {W}")
    print()
    print(f"{'ID':>4} | {'Chi phí':>8} | {'Lợi ích':>8} | {'Chọn?':>6}")
    print("-" * 40)
    chosen_set = set(chosen_indices)
    for idx in range(len(weights)):
        mark = "  X" if idx in chosen_set else "   "
        print(f"{item_ids[idx]:>4} | {weights[idx]:>8} | {values[idx]:>8} | {mark:>6}")

    total_cost = sum(weights[i] for i in chosen_indices)
    print("-" * 40)
    print(f"Số dự án được chọn      : {len(chosen_indices)}/{len(weights)}")
    print(f"Tổng chi phí sử dụng     : {total_cost} / {W} "
          f"(còn dư {W - total_cost})")
    print(f"TỔNG LỢI ÍCH TỐI ƯU      : {best_value}")
    print()
    print("Danh sách ID dự án được chọn:")
    print("  " + ", ".join(str(item_ids[i]) for i in chosen_indices))
    print("=" * 65)


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Cho phép truyền đường dẫn CSV qua dòng lệnh, ví dụ:
    #   python3 knapsack_dp_solution.py /mnt/user-data/uploads/dt02_knapsack_medium.csv
    # Mặc định vẫn dùng bộ dữ liệu small nếu không truyền tham số.
    CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/dt02_knapsack_small.csv"

    item_ids, weights, values, W = load_knapsack_csv(CSV_PATH)

    t0 = time.perf_counter()
    best_value, chosen_indices = knapsack_dp_full(weights, values, W)
    elapsed = time.perf_counter() - t0

    print_solution(item_ids, weights, values, W, best_value, chosen_indices)
    print(f"Thời gian chạy DP: {elapsed:.6f} giây "
          f"(n={len(weights)}, W={W}, số ô bảng DP = {len(weights)*(W+1):,})")
		  
		  
		  