```
 $ ncc_0.1_win\ncc test.tflite test.kmodel -i tflite -o k210model —dataset train_img
```
### 2# 下载工具 
   下载此工程，在工程根目录下将[train_ann.zip]和[train_img.zip]解压到当前文件夹

   下载ncc工具箱：网盘下载：https://pan.baidu.com/s/1NT2tG4Rv2YJyjOKRh-3t4w  提取码：z9fr
   
   csdn下载：https://download.csdn.net/download/qq_40508193/12261414
   
   将[ncc_0.1_win.zip]放置在工程根目录，解压到当前文件夹
        
-*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- 



    
### 7# 转换成Kmodel：
   [(yolo) $ ncc_0.1_win\ncc test.tflite test.kmodel -i tflite -o k210model --dataset train_img]

   转换完成根目录会出现test.kmodel，即可烧录进k210中运行
   
-*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- 

   
### 8# 运行
   maixpy程序见(maixpy_code)文件夹，如有修改configs记得修改对应的archor、图像大小(224*224)、lable
