#include "util.h"

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        printf("第二引数がありません\n");
        abort();
    }

    float *train_x = NULL;
    unsigned char *train_y = NULL;
    int train_count = -1;

    float *test_x = NULL;
    unsigned char *test_y = NULL;
    int test_count = -1;

    int width = -1;
    int height = -1;

    load_mnist(&train_x, &train_y, &train_count,
               &test_x, &test_y, &test_count,
               &width, &height);

    srand(time(NULL));

    //test_learning_1(train_x, train_y, test_x, test_y, width, height, 10, 100, train_count, test_count, 0.1);

    //float *y = malloc(sizeof(float) * 10);

    // 3層パーセプトロンの推論テスト
    /*
    int sum = 0;
    float loss = 0;
    for (int i = 0; i < test_count; i++)
    {
        if (inference6(A1_784_50_100_10, b1_784_50_100_10, A2_784_50_100_10, b2_784_50_100_10, A3_784_50_100_10, b3_784_50_100_10,
               test_x + 784*i, y) == test_y[i])
        {
            sum++;
        }
        loss += cross_entropy_error(y, test_y[i]);
    }
    printf("loss : %f, accuracy : %f%%\n", loss / test_count, sum * 100.0 / test_count);
    */

    // パラメータ
    float *A1 = malloc(sizeof(float) * HID_1 * IN);
    float *b1 = malloc(sizeof(float) * HID_1);
    float *A2 = malloc(sizeof(float) * HID_2 * HID_1);
    float *b2 = malloc(sizeof(float) * HID_2);
    float *A3 = malloc(sizeof(float) * OUT * HID_2);
    float *b3 = malloc(sizeof(float) * OUT);
    // パラメータの初期化
    
    rand_normal(HID_1 * IN, A1, IN);
    rand_normal(HID_1, b1, HID_1);
    rand_normal(HID_2 * HID_1, A2, HID_1);
    rand_normal(HID_2, b2, HID_2);
    rand_normal(OUT * HID_2, A3, HID_2);
    rand_normal(OUT, b3, OUT);
    /*
    rand_init(HID_1 * IN, A1);
    rand_init(HID_1, b1);
    rand_init(HID_2 * HID_1, A2);
    rand_init(HID_2, b2);
    rand_init(OUT * HID_2, A3);
    rand_init(OUT, b3);
    */

    test_learning_3(A1, b1, A2, b2, A3, b3, train_x, train_y, test_x, test_y, width, height, 10, 100, train_count, test_count, atof(argv[1]));
}