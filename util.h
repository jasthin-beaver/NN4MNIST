#include <time.h>
#include "nn.h"

// 出力次元数
#define OUT 10
// 入力次元数
#define IN 784
// 中間層次元数
#define HID_1 50
#define HID_2 100

void init(int n, float x, float *y)
{
    /* 配列yの全要素を値xで初期化 */
    for (int i = 0; i < n; i++)
        y[i] = x;
}

void add(int n, const float *x, float *y)
{
    /* 配列xと配列yの要素同士を足し合わせてyに格納 */
    for (int i = 0; i < n; i++)
        y[i] += x[i];
}

void scale(int n, float x, float *y)
{
    /* 配列yの各要素をx倍する */
    for (int i = 0; i < n; i++)
        y[i] *= x;
}

void print(int m, int n, const float *x)
{
    /* 配列の表示 */
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%.4f ", x[i * n + j]);
        }
        printf("\n");
    }
}

void print_int(int m, int n, const int *x)
{
    /* 配列の表示 */
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%d ", x[i * n + j]);
        }
        printf("\n");
    }
}

void fc(int m, int n, const float *x, const float *A, const float *b, float *y)
{
    /* アフィン変換 */
    init(m, 0, y);
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            y[i] += A[i * n + j] * x[j];
        }
        y[i] += b[i];
    }
}

void relu(int n, const float *x, float *y)
{
    /* Relu関数 */
    for (int i = 0; i < n; i++)
    {
        y[i] = x[i] > 0 ? x[i] : 0;
    }
}

void softmax(int n, const float *x, float *y)
{
    /* ソフトマックス関数 */
    // 最大値を見つける
    float x_max = x[0];
    for (int i = 1; i < n; i++)
    {
        if (x_max < x[i])
            x_max = x[i];
    }
    // ソフトマックス
    float sum = 0;
    for (int i = 0; i < n; i++)
    {
        y[i] = expf(x[i] - x_max);
        sum += y[i];
    }
    scale(n, 1 / sum, y);
}

int max_index(float *y)
{
    /* 配列yの最大値をとる添字を返却 */
    float max_value = y[0];
    int max_i = 0;
    for (int i = 1; i < OUT; i++)
    {
        if (max_value < y[i])
        {
            max_value = y[i];
            max_i = i;
        }
    }
    return max_i;
}

int inference3(const float *A, const float *b, const float *x, float *y)
{
    // 単層パーセプトロンによる推論
    init(OUT, 0, y);

    fc(OUT, IN, x, A, b, y);
    relu(OUT, y, y);
    softmax(OUT, y, y);

    // 最大となるインデックス探し
    return max_index(y);
}

void softmaxwithloss_bwd(int n, const float *y, unsigned char t, float *dEdx)
{
    /* ソフトマックスの逆伝搬 */
    for (int i = 0; i < n; i++)
    {
        dEdx[i] = (t == i) ? y[i] - 1 : y[i];
    }
}

void relu_bwd(int n, const float *x, const float *dEdy, float *dEdx)
{
    /* ReLUの逆伝搬 */
    for (int i = 0; i < n; i++)
    {
        dEdx[i] = x[i] > 0 ? dEdy[i] : 0;
    }
}

void fc_bwd(int m, int n, const float *x, const float *dEdy, const float *A,
            float *dEdA, float *dEdb, float *dEdx)
{
    /* アフィンの逆伝搬 */
    init(n, 0, dEdx);
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            dEdA[i * n + j] = dEdy[i] * x[j];
            dEdx[j] += A[i * n + j] * dEdy[i];
        }
        dEdb[i] = dEdy[i];
    }
}

void backward3(const float *A, const float *b, const float *x, unsigned char t,
               float *y, float *dEdA, float *dEdb)
{
    /* 単層パーセプトロンの誤差逆伝播 */
    float *y_fc = malloc(sizeof(float) * OUT);
    float *y_relu = malloc(sizeof(float) * OUT);
    float *dEdy = malloc(sizeof(float) * OUT);
    float *dEdx = malloc(sizeof(float) * IN);

    fc(OUT, IN, x, A, b, y_fc);
    printf("y_fc\n");
    print(1, OUT, y_fc);
    relu(OUT, y_fc, y_relu);
    printf("y_relu\n");
    print(1, OUT, y_relu);
    softmax(OUT, y_relu, y);
    printf("y\n");
    print(1, OUT, y);

    softmaxwithloss_bwd(OUT, y, t, dEdy);
    printf("dEdy\n");
    print(1, OUT, dEdy);
    relu_bwd(OUT, y_fc, dEdy, dEdy);
    fc_bwd(OUT, IN, x, dEdy, A, dEdA, dEdb, dEdx);

    free(y_fc);
    free(y_relu);
    free(dEdy);
    free(dEdx);
}

void swap(int *x, int *y)
{
    int temp = *x;
    *x = *y;
    *y = temp;
}

void shuffle(int n, int *x)
{
    /* 配列要素をシャッフル */
    int j;
    for (int i = 0; i < n; i++)
        x[i] = i;
    for (int i = 0; i < n; i++)
    {
        j = rand() % n;
        swap(&x[i], &x[j]);
    }
}

float cross_entropy_error(const float *y, int t)
{
    /* クロスエントロピー損失 */
    return -logf(y[t] + (1e-7));
}

void rand_init(int n, float *x)
{
    /* 配列の各要素を[-1,1]の乱数で初期化 */
    for (int i = 0; i < n; i++)
    {
        x[i] = 2 * (rand() / (double)RAND_MAX) - 1;
    }
}

void init_index(int n, int *index)
{
    for (int i = 0; i < n; i++)
    {
        index[i] = i;
    }
}

int inference6(const float *A_1, const float *b_1, const float *A_2, const float *b_2, const float *A_3, const float *b_3, const float *x, float *y)
{
    /* 3層パーセプトロンの推論 */
    float *y1 = malloc(sizeof(float) * HID_1);
    float *y2 = malloc(sizeof(float) * HID_2);
    init(HID_1, 0, y1);
    init(HID_2, 0, y2);
    fc(HID_1, IN, x, A_1, b_1, y1);
    relu(HID_1, y1, y1);
    fc(HID_2, HID_1, y1, A_2, b_2, y2);
    relu(HID_2, y2, y2);
    fc(OUT, HID_2, y2, A_3, b_3, y);
    //relu(OUT, y, y);
    softmax(OUT, y, y);

    free(y1);
    free(y2);

    return max_index(y);
}

void backward6(const float *A_1, const float *b_1, const float *A_2, const float *b_2, const float *A_3, const float *b_3,
               const float *x, unsigned char t, float *y, float *dEdA1, float *dEdb1, float *dEdA2, float *dEdb2, float *dEdA3, float *dEdb3)
{
    /* 3層パーセプトロンの逆伝搬 */
    // 出力
    float *y1 = malloc(sizeof(float) * HID_1);
    float *y2 = malloc(sizeof(float) * HID_2);
    float *y3 = malloc(sizeof(float) * OUT);

    // 勾配
    float *dEdy1 = malloc(sizeof(float) * HID_1);
    float *dEdy2 = malloc(sizeof(float) * HID_2);
    float *dEdy3 = malloc(sizeof(float) * OUT);
    float *dEdx = malloc(sizeof(float) * IN);

    // 順伝搬
    fc(HID_1, IN, x, A_1, b_1, y1);
    relu(HID_1, y1, y1);
    fc(HID_2, HID_1, y1, A_2, b_2, y2);
    relu(HID_2, y2, y2);
    fc(OUT, HID_2, y2, A_3, b_3, y3);
    //relu(OUT, y3, y3);
    softmax(OUT, y3, y);

    // 逆伝搬
    softmaxwithloss_bwd(OUT, y, t, dEdy3);
    //relu_bwd(OUT, y3, dEdy3, dEdy3);
    fc_bwd(OUT, HID_2, y2, dEdy3, A_3, dEdA3, dEdb3, dEdy2);
    relu_bwd(HID_2, y2, dEdy2, dEdy2);
    fc_bwd(HID_2, HID_1, y1, dEdy2, A_2, dEdA2, dEdb2, dEdy1);
    relu_bwd(HID_1, y1, dEdy1, dEdy1);
    fc_bwd(HID_1, IN, x, dEdy1, A_1, dEdA1, dEdb1, dEdx);

    // 必要のないものは解放
    free(y1);
    free(y2);
    free(y3);
    free(dEdy1);
    free(dEdy2);
    free(dEdy3);
    free(dEdx);
}

void test_learning_1(float *train_x, unsigned char *train_y, float *test_x, unsigned char *test_y, int width, int height,
                     int num_epoch, int batch_size, int train_count, int test_count, float lr)
{
    /* 単層パーセプトロンの学習 */
    // パラメータ
    float *A = malloc(sizeof(float) * OUT * IN);
    float *b = malloc(sizeof(float) * OUT);
    rand_init(OUT * IN, A);
    rand_init(OUT, b);
    // SGDのための変数
    int *index = malloc(sizeof(int) * train_count);
    int lower, upper;
    float s;
    // 勾配格納用配列
    float *dEdA = malloc(sizeof(float) * OUT * IN);
    float *dEdb = malloc(sizeof(float) * OUT);
    float *dEdA_mean = malloc(sizeof(float) * OUT * IN);
    float *dEdb_mean = malloc(sizeof(float) * OUT);
    // 出力
    float *y = malloc(sizeof(float) * OUT);
    // testにおける総正解数と損失
    int sum;
    float loss;

    // 配列indexに添字を代入
    init_index(train_count, index);

    // 学習
    for (int epoch = 0; epoch < num_epoch; epoch++)
    {
        // indexのシャッフル
        //init_index(train_count, index);
        shuffle(train_count, index);

        for (int n = 0; n < train_count / batch_size; n++)
        {
            // ミニバッチ学習前に平均勾配は0で初期化
            init(OUT * IN, 0, dEdA_mean);
            init(OUT, 0, dEdb_mean);
            lower = n * batch_size;
            upper = (n + 1) * batch_size < train_count ? (n + 1) * batch_size : train_count;
            for (int i = lower; i < upper; i++)
            {
                backward3(A, b, train_x + IN * index[i], train_y[index[i]], y, dEdA, dEdb);
                add(OUT * IN, dEdA, dEdA_mean);
                add(OUT, dEdb, dEdb_mean);
            }
            s = -lr / (upper - lower);
            scale(OUT * IN, s, dEdA_mean);
            scale(OUT, s, dEdb_mean);
            add(OUT * IN, dEdA_mean, A);
            add(OUT, dEdb_mean, b);
        }
        //print(1, M, b);

        // 評価
        sum = 0;
        loss = 0;
        for (int i = 0; i < test_count; i++)
        {
            if (inference3(A, b, test_x + i * width * height, y) == test_y[i])
            {
                sum++;
            }
            loss += cross_entropy_error(y, test_y[i]);
        }
        printf("Epoch[%d/%d] ### loss : %.4f, accuracy : %.4f%%\n", epoch + 1, num_epoch, loss / test_count, sum * 100.0 / test_count);
    }

    // 解放
    free(dEdA);
    free(dEdb);
    free(dEdA_mean);
    free(dEdb_mean);
    free(A);
    free(b);
    free(index);
    free(y);
}

float Uniform(){
    /* 一様乱数を返却 */
    return (float)rand()/((float)RAND_MAX+1.0);
}

void rand_normal(int n, float* x, int value){
    /* ガウス分布に従う変数で初期化(Box-Muller法) Heの初期化 */
    for(int i=0;i<n;i++){
        x[i] = sqrt(2.0 / value) * sqrt( -2.0*logf(Uniform()) ) * sinf( 2.0*M_PI*Uniform() );
    }
 }

void test_learning_3(float* A1, float* b1, float* A2, float* b2, float* A3, float* b3, float *train_x, unsigned char *train_y, float *test_x, unsigned char *test_y, int width, int height,
                     int num_epoch, int batch_size, int train_count, int test_count, float lr)
{
    /* 3層パーセプトロンの学習 */
    printf("エポック数 : %d\nバッチ数 : %d\n学習率 : %.3f\n", num_epoch, batch_size, lr);

    // 勾配
    float *dEdA1 = malloc(sizeof(float) * HID_1 * IN);
    float *dEdb1 = malloc(sizeof(float) * HID_1);
    float *dEdA2 = malloc(sizeof(float) * HID_2 * HID_1);
    float *dEdb2 = malloc(sizeof(float) * HID_2);
    float *dEdA3 = malloc(sizeof(float) * OUT * HID_2);
    float *dEdb3 = malloc(sizeof(float) * OUT);
    float *dEdA1_mean = malloc(sizeof(float) * HID_1 * IN);
    float *dEdb1_mean = malloc(sizeof(float) * HID_1);
    float *dEdA2_mean = malloc(sizeof(float) * HID_2 * HID_1);
    float *dEdb2_mean = malloc(sizeof(float) * HID_2);
    float *dEdA3_mean = malloc(sizeof(float) * OUT * HID_2);
    float *dEdb3_mean = malloc(sizeof(float) * OUT);

    // 出力
    float *y = malloc(sizeof(float) * OUT);

    // SGD用の変数
    int *index = malloc(sizeof(int) * train_count);
    init_index(train_count, index);
    int upper, lower;
    float param;

    // 推論用変数
    int sum;
    float loss;

    // 学習
    for (int epoch = 0; epoch < num_epoch; epoch++)
    {
        // シャッフル
        shuffle(train_count, index);

        // ミニバッチ学習
        for (int n = 0; n < train_count / batch_size; n++)
        {
            // 勾配の初期化
            init(HID_1 * IN, 0, dEdA1_mean);
            init(HID_1, 0, dEdb1_mean);
            init(HID_2 * HID_1, 0, dEdA2_mean);
            init(HID_2, 0, dEdb2_mean);
            init(OUT * HID_2, 0, dEdA3_mean);
            init(OUT, 0, dEdb3_mean);
            lower = n * batch_size;
            upper = (n + 1) * batch_size < train_count ? (n + 1) * batch_size : train_count;
            for (int i = lower; i < upper; i++)
            {
                backward6(A1, b1, A2, b2, A3, b3, train_x + IN * index[i], train_y[index[i]], y, dEdA1, dEdb1, dEdA2, dEdb2, dEdA3, dEdb3);
                add(HID_1 * IN, dEdA1, dEdA1_mean);
                add(HID_1, dEdb1, dEdb1_mean);
                add(HID_2 * HID_1, dEdA2, dEdA2_mean);
                add(HID_2, dEdb2, dEdb2_mean);
                add(OUT * HID_2, dEdA3, dEdA3_mean);
                add(OUT, dEdb3, dEdb3_mean);
            }
            param = -lr / (upper - lower);
            scale(HID_1 * IN, param, dEdA1_mean);
            scale(HID_1, param, dEdb1_mean);
            scale(HID_2 * HID_1, param, dEdA2_mean);
            scale(HID_2, param, dEdb2_mean);
            scale(OUT * HID_2, param, dEdA3_mean);
            scale(OUT, param, dEdb3_mean);

            add(HID_1 * IN, dEdA1_mean, A1);
            add(HID_1, dEdb1_mean, b1);
            add(HID_2 * HID_1, dEdA2_mean, A2);
            add(HID_2, dEdb2_mean, b2);
            add(OUT * HID_2, dEdA3_mean, A3);
            add(OUT, dEdb3_mean, b3);
        }
        print(1, 10, b3);

        // 評価
        sum = 0;
        loss = 0;
        for (int i = 0; i < test_count; i++)
        {
            if (inference6(A1, b1, A2, b2, A3, b3, test_x + IN * i, y) == test_y[i])
                sum++;
            loss += cross_entropy_error(y, test_y[i]);
        }
        printf("Epoch[%d/%d] ### loss : %.4f, accuracy : %.4f%%\n", epoch + 1, num_epoch, loss / test_count, sum * 100.0 / test_count);
    }

    // 解放
    free(dEdA1);
    free(dEdA2);
    free(dEdA3);
    free(dEdb1);
    free(dEdb2);
    free(dEdb3);
    free(dEdA1_mean);
    free(dEdA2_mean);
    free(dEdA3_mean);
    free(dEdb1_mean);
    free(dEdb2_mean);
    free(dEdb3_mean);
    free(A1);
    free(A2);
    free(A3);
    free(b1);
    free(b2);
    free(b3);
    free(index);
    free(y);
}