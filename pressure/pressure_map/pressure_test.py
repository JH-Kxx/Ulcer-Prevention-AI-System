import serial
import numpy as np
import cv2

# -----------------------------
# 설정
# -----------------------------

WIDTH = 400
HEIGHT = 800

SIGMA = 45

MAX_HEAT = 1000

PRESSURE_THRESHOLD = 350

# -----------------------------
# Serial
# -----------------------------

ser1 = serial.Serial(
    'COM3',
    115200,
    timeout=0.01
)

ser2 = serial.Serial(
    'COM4',
    115200,
    timeout=0.01
)

ser1.reset_input_buffer()
ser2.reset_input_buffer()

# -----------------------------
# 마지막 정상 데이터
# -----------------------------

last_values1 = [0] * 16
last_values2 = [0] * 16

# -----------------------------
# Window
# -----------------------------

cv2.namedWindow(
    "Pressure Heatmap",
    cv2.WINDOW_NORMAL
)

# -----------------------------
# Gaussian 미리 계산
# -----------------------------

Y, X = np.mgrid[0:HEIGHT, 0:WIDTH]

cell_w = WIDTH // 4
cell_h = HEIGHT // 8

gaussian_maps = []

for idx in range(32):

    row = idx // 4
    col = idx % 4

    cx = int(
        (col + 0.5) * cell_w
    )

    cy = int(
        (row + 0.5) * cell_h
    )

    g = np.exp(
        -(
            (X - cx) ** 2 +
            (Y - cy) ** 2
        ) / (2 * SIGMA ** 2)
    ).astype(np.float32)

    gaussian_maps.append(g)

# -----------------------------
# Main Loop
# -----------------------------

while True:

    try:

        line1 = ser1.readline().decode(
            errors='ignore'
        ).strip()

        if line1:

            parts1 = line1.split(',')

            if len(parts1) == 16:

                last_values1 = [
                    int(x)
                    for x in parts1
                ]

    except:
        pass

    try:

        line2 = ser2.readline().decode(
            errors='ignore'
        ).strip()

        if line2:

            parts2 = line2.split(',')

            if len(parts2) == 16:

                last_values2 = [
                    int(x)
                    for x in parts2
                ]

    except:
        pass

    # =====================================================
    # 총 32개
    # =====================================================

    values = (
        last_values1 +
        last_values2
    )

    # =====================================================
    # Heatmap 생성
    # =====================================================

    heatmap = np.zeros(
        (HEIGHT, WIDTH),
        dtype=np.float32
    )

    for idx in range(32):

        value = values[idx]

        if value < PRESSURE_THRESHOLD:
            continue

        heatmap += (
            gaussian_maps[idx]
            * value
        )

    # =====================================================
    # 정규화
    # =====================================================

    heatmap = np.clip(
        heatmap / MAX_HEAT * 255,
        0,
        255
    ).astype(np.uint8)

    # =====================================================
    # Heatmap
    # =====================================================

    color_map = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    # =====================================================
    # Color Bar
    # =====================================================

    bar_width = 90

    bar_gray = np.zeros(
        (HEIGHT, 30),
        dtype=np.uint8
    )

    for y in range(HEIGHT):

        value = int(
            255 -
            (y / HEIGHT) * 255
        )

        bar_gray[y, :] = value

    color_bar = cv2.applyColorMap(
        bar_gray,
        cv2.COLORMAP_JET
    )

    bar_panel = np.full(
        (HEIGHT, bar_width, 3),
        255,
        dtype=np.uint8
    )

    bar_panel[:, 10:40] = color_bar

    cv2.putText(
        bar_panel,
        "ADC",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2
    )

    ticks = [
        350,
        500,
        750,
        1000
    ]

    for tick in ticks:

        y = int(
            HEIGHT -
            (tick / MAX_HEAT)
            * HEIGHT
        )

        y = np.clip(
            y,
            10,
            HEIGHT - 10
        )

        cv2.line(
            bar_panel,
            (40, y),
            (55, y),
            (0, 0, 0),
            2
        )

        cv2.putText(
            bar_panel,
            str(tick),
            (58, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1
        )

    # =====================================================
    # 최종 합치기
    # =====================================================

    final_img = np.hstack([
        color_map,
        bar_panel
    ])

    cv2.imshow(
        "Pressure Heatmap",
        final_img
    )

    ser1.reset_input_buffer()
    ser2.reset_input_buffer()

    if cv2.waitKey(1) == 27:
        break

# -----------------------------
# 종료
# -----------------------------

ser1.close()
ser2.close()

cv2.destroyAllWindows()