# 数据准备

三期数据是用PPOCRLabel标注的。原始标注数据放在 D:\datasets\sanqi\ 目录下面.

之后这些数据用PPOCRLabel提供的工具，gen_ocr_train_val_test.py，split，split好的数据放在det和rec目录下面。

```
python gen_ocr_train_val_test.py --trainValTestRatio 6:2:2 --datasetRootPath D:\datasets\sanqi\ --detRootPath D:\datasets\sanqi\det --recRootPath D:\datasets\sanqi\rec --detLabelFileName Label.txt --recLabelFileName rec_gt.txt --recImageDirName crop_img
```

为了提高detection的速度，尝试了reduce图像resolution到1/2,1/3和1/4。分别放在det_1_1，det_1_2，det_1_3，det_1_4目录。

```
python sanqi\det\reduce_resolution.py 
```
Note, 这个script只从D:\datasets\sanqi\det目录读取，并把结果保存在D:\datasets\sanqi\det_reduced目录下面。然后我们在rename成det_1_1，det_1_2，det_1_3，det_1_4等。

# 训练概述

所有有关三期的训练都在sanqi目录下面。

- pretrain_models目录下面是从PPOCR下载的pretrained模型
- sanqi_dict.txt是字典
- 训练结果放在output目录下面
- 所以测试过的模型放在output\inference目录下面。
    - Detection模型
        - ch_PP-OCRv3_det_infer （PPOCR pretrained 模型）
        - db_mv3_det_sanqi （det_mv3_db.yml refit出的模型）
            - 1_1 (in original 1:1 resolution)
            - 3_1 (1/3 of the original resolution)
    - Recognition模型
        - ch_PP-OCRv3_rec_infer  （PPOCR pretrained 模型）
        - en_PP-OCRv3_rec_sanqi （en_PP-OCRv3_rec.yml refit出的模型）

## Get Pretrained model command
```
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_slim_distill_train.tar -OutFile ./en_PP-OCRv3_det_slim_distill_train.tar
tar xvf en_PP-OCRv3_det_slim_distill_train.tar
```

# Detection模型训练

总共124个training images。按照6:2:2分为train，val，test sets。train有75个image；val有25个image；test有24个image。

## det_mv3_db（det_mv3_db.yml）的方式。
最优训练模型在`./sanqi/output/det/db_mv3`目录下面。
```
python tools/train.py -c ./sanqi/det/det_mv3_db.yml
```
## ch_PP-OCRv3_det_cml（ch_PP-OCRv3_det_cml.yml）的方式。
最优训练模型在`./sanqi/output/det/ch_PP-OCRv3_det`目录下面。
```
python tools/train.py -c ./sanqi/det/ch_PP-OCRv3_det_cml.yml
```


Evaluation: The performance is not excellent yet. Inceasing training samples may a solution
```
python tools/eval.py -c sanqi/det/det_mv3_db.yml -o Global.checkpoints=.\sanqi\output\det\db_mv3\best_accuracy  Eval.dataset.data_dir=D:\datasets\sanqi\det Eval.dataset.label_file_list=["D:\datasets\sanqi\det\test.txt"] PostProcess.box_thresh=0.6 PostProcess.unclip_ratio=1.5
```
```
python tools/eval.py -c sanqi/det/ch_PP-OCRv3_det_cml.yml -o Global.checkpoints=.\sanqi\output\det\ch_PP-OCRv3_det\best_accuracy  Eval.dataset.data_dir=D:\datasets\sanqi\det Eval.dataset.label_file_list=["D:\datasets\sanqi\det\test.txt"] PostProcess.box_thresh=0.6 PostProcess.unclip_ratio=1.5
```

Infer: Run this command can generate the detection results of all test images, so we can quickly get an impression of the performance. The resulting images are saved at `.\sanqi\output\det\det_results` folder.
```
python tools/infer_det.py -c sanqi/det/det_mv3_db.yml -o Global.pretrained_model=.\sanqi\output\det\db_mv3\best_accuracy  Global.infer_img=D:\datasets\sanqi\det\test\ Global.save_res_path=.\sanqi\output\det\predicts_db.txt
```

Export Model: Exported的模型在D:\github\PaddleOCR\sanqi\output\inference\db_mv3_det_sanqi目录下面。
```
python tools/export_model.py -c ./sanqi/det/det_mv3_db.yml -o Global.pretrained_model=.\sanqi\output\det\db_mv3\best_accuracy  Global.save_inference_dir=.\sanqi\output\inference\db_mv3_det_sanqi
```
```
python tools/export_model.py -c ./sanqi/det/ch_PP-OCRv3_det_cml.yml -o Global.pretrained_model=.\sanqi\output\det\ch_PP-OCRv3_det\best_accuracy  Global.save_inference_dir=.\sanqi\output\det_ch_PP-OCRv3_det_slim
```
```
python tools/export_model.py -c ./configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml -o Global.pretrained_model=.\sanqi\pretrain_models\en_PP-OCRv3_det_slim_distill_train\best_recall  Global.save_inference_dir=.\sanqi\output\det_en_PP-OCRv3_det_slim
```

Infer with Exported Model: Similarly, run this command, we can quickly generate the detection results of all test images for getting a quick impression of the performance. The generated results are saved at `./inference_results` folder.
The results from exported model (predict_det.py) seem better than training model (infer_det.py).
```
python tools/infer/predict_det.py --image_dir=D:\datasets\sanqi\det\test\ --det_model_dir=.\sanqi\output\inference\db_mv3_det_sanqi
```

The list of inference model that we have:
* ch_PP-OCRv3_det_infer: pre-trained PPOCR detection model, it is for general text detection. Certainly not good for Sanqi text detection
* db_mv3_det_sanqi_1_1: Origianl image, speed is slow
* db_mv3_det_sanqi_1_2: 1/4 of the original image
* db_mv3_det_sanqi_1_3: 1/9 of the original image. 

## Performance comparison
db_mv3_det_sanqi_1_1: val precison 0.6364; val recall 0.8630; test precision 0.3609; test recall: 0.6957;
db_mv3_det_sanqi_1_2: val precison 0.6058; val recall 0.8630; test precision 0.4423; test recall: 0.6666;
db_mv3_det_sanqi_1_2: val precison 0.5652; val recall 0.8904; test precision 0.4091; test recall: 0.6521;
 ch_PP-OCRv3_det_cml: val precison 0.5161; val recall 0.8767; test precision 0.4365; test recall: 0.7971;

## Inference speed comparison
|   Infereence Speed        | 1:1   | 1:2   | 1:3   | 1:4   |
| -------------             | ----- | ----- | ----- | ----- |
| ch_PP-OCRv3_det_infer     | 475ms | 220ms | 110ms | NoDet |
| db_mv3_det_sanqi_1_1      | 775ms | 330ms | 160ms | NoDet |
| db_mv3_det_sanqi_1_2      | 750ms | 340ms | 160ms | NoDet |
| db_mv3_det_sanqi_1_3      | 725ms | 340ms | 150ms | NoDet |
|  ch_PP-OCRv3_det_cml      | 725ms | ----- | ----- | ----- |

## Some conclusions

ch_PP-OCRv3_det_infer (pretrained model) is the fastest, and can detect general texts better. But, it has issues detecting Sanqi texts, especially dotted texts and stencil texts.

The other 3 are compariable, not working when reduce to larger than 1/3.

For now, I am going to use db_mv3_det_sanqi_1_1 detect at original resolution. The speed is a big issue.

## To Do:
Speed is a big issue
* refit ch_PP-OCRv3_det_infer model
* refit rec model with smaller image size
* Use knowledge distillation: https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/knowledge_distillation.md
* Different model comparison: https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/models_list_en.md

It seems that using ch_PP-OCRv3_det_cml model, compariable performance is achieved and the processing time id reduce from ~800ms to ~500ms.

# Recognition模型训练

Training: 目前选用en_PP-OCRv3_rec.yml作为训练config。总共375个croped word image。按照6:2:2分为train，val，test sets。train有225个image；val有75个image；test有75个image。最优训练模型在sainqi/output/rec0目录下面。batch size设为16。在val上accuracy是0.9687498486328361（62 out of 64）。在test上accuracy是0.9594593298027932 （71 out of 74）。
```
python tools/train.py -c ./sanqi/rec/en_PP-OCRv3_rec.yml
```

Evaluation: performance looks good, but speed may be an issue
```
python tools/eval.py -c sanqi/rec/en_PP-OCRv3_rec.yml -o Global.checkpoints=.\sanqi\output\rec0\en_rec_ppocr_v3_rec\best_accuracy  Eval.dataset.data_dir=D:\datasets\sanqi\rec Eval.dataset.label_file_list=["D:\datasets\sanqi\rec\test.txt"]
```

Infer: Run this command can generate the recognition results of all test images, so we can quickly get an impression of the performance. The resulting images are saved at `.\sanqi\output\rec\predicts_ppocrv3_en.txt`.
```
python tools/infer_rec.py -c sanqi/rec/en_PP-OCRv3_rec.yml -o Global.pretrained_model=.\sanqi\output\rec0\en_rec_ppocr_v3_rec\best_accuracy  Global.infer_img=D:\datasets\sanqi\rec\test
```

Export Model: Exported的模型在D:\github\PaddleOCR\sanqi\output\inference\en_PP-OCRv3_rec_sanqi目录下面。
```
python tools/export_model.py -c ./sanqi/rec/en_PP-OCRv3_rec.yml -o Global.pretrained_model=.\sanqi\output\rec0\en_rec_ppocr_v3_rec\best_accuracy  Global.save_inference_dir=.\sanqi\output\inference\en_PP-OCRv3_rec_sanqi/
```

Infer with Exported Model:
```
python tools/infer/predict_rec.py --image_dir="D:\datasets\sanqi\rec\test" --rec_model_dir=".\sanqi\output\inference\en_PP-OCRv3_rec_sanqi" --rec_image_shape="3, 48, 320" --rec_char_dict_path=.\sanqi\sanqi_dict.txt
```

# TODO:
* Recognition speed may also need to improve. The goal is to reach ~250ms. We can do that by reducing `image_shape`

# 其它
PaddleOCR repo中的applications目录下面有一个`包装生产日期识别`的应用，值得看一下。

