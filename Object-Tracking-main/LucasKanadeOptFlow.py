import numpy as np
import cv2
from segRGB import KmeansSeg

def optical_flow(frame_t, frame_t_1, window_size): #window_size u,v 구하기위한 근접픽셀수 설정

    corners = cv2.goodFeaturesToTrack(frame_t, 0, 0.01, 0.1) #코너 찾기함수 (이미지,최대코너갯수(0=무한대),코너점 결정을 위한 값,코너점 사이의 최소 거리)

    frame_t = frame_t / 255
    frame_t_1 = frame_t_1 / 255

    kernel_x = np.array([[-1, 1], [-1, 1]]) #edge detection을 위한 Kernel들
    kernel_y = np.array([[-1, -1], [1, 1]])
    kernel_t = np.array([[1, 1], [1, 1]])

    fx = cv2.filter2D(frame_t, -1, kernel_x)              #X,Y,T 의 gradient 구하기
    fy = cv2.filter2D(frame_t, -1, kernel_y)
    ft = cv2.filter2D(frame_t_1, -1, kernel_t) - cv2.filter2D(frame_t, -1, kernel_t)


    u = np.zeros(frame_t.shape)
    v = np.zeros(frame_t.shape)

    n = int(window_size / 2)

    for feature in corners:
            j, i = feature.ravel()
            i, j = int(i), int(j)

            I_x = fx[i-n:i+n+1, j-n:j+n+1].flatten()  # 이웃픽셀 들 추가
            I_y = fy[i-n:i+n+1, j-n:j+n+1].flatten()
            I_t = ft[i-n:i+n+1, j-n:j+n+1].flatten()

            b = np.reshape(I_t, (I_t.shape[0], 1))
            A = np.vstack((I_x, I_y)).T  #수직으로 행렬 결합

            V = np.matmul(np.linalg.pinv(A), b)     #/matmul =행렬곱 함수/ linalg.pinv 의사 역행렬

            u[i, j] = V[0][0]
            v[i, j] = V[1][0]
 
    return (u, v)


def getresultimg(frame, U, V, output):

    line_color = (0, 255, 0)

    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            u, v = U[i][j], V[i][j]

            if u and v:
                frame = cv2.arrowedLine( frame, (i, j), (int(round(i+u)), int(round(j+v))), line_color, thickness=1)
    cv2.imwrite(output, frame)


img1 = cv2.imread("./Inputs/1.png")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

img2 = cv2.imread("./Inputs/2.png")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

U, V = optical_flow( img1, img2, 3)

img2 = cv2.cvtColor( img2, cv2.COLOR_GRAY2RGB)
#getresultimg(img2, U, V, './Results/Result3.png')

inputimg = cv2.imread("./Inputs/1.png")

seg = KmeansSeg()

rgbout = seg.segmentation(inputimg, 4)
cv2.imwrite('./Results/k4.png', rgbout)
