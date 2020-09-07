# mAP - By: yiyuan
# this file is used to generate .txt file for mAP calculation

import image, time, lcd
import KPU as kpu

lcd.init(freq=15000000)
classes = ['black+black+red', 'black+black+silver', 'black+white+red', 'black+white+silver', 'white+black+red', 'white+black+silver', 'white+white+red', 'white+white+silver']
task = kpu.load(0x300000)
anchor = [1.51,2.43, 1.83,1.54, 2.26,3.08, 2.59,2.07, 3.75,3.41]
a = kpu.init_yolo2(task, 0.05, 0.3, 5, anchor)

for pic in list(os.listdir('/sd/pics')):
    img = image.Image('/sd/pics/' + pic)
    print(pic[:-4])
    img.pix_to_ai()
    code = kpu.run_yolo2(task, img)
    #del(img)
    #print(code)
    if code:
        ff1 = open('/sd/'+ pic[:-4] + '.txt', 'w')
        for i in code:
            print('i:', i)
            print('rect:', i.rect())
            print(classes[i.classid()], i.value(),i.x(), i.y(), i.w(), i.h())
            ff1.write("%s %s %s %s %s %s\n" % (classes[i.classid()], i.value(),i.x(), i.y(), i.w(), i.h()))
        ff1.close()

a = kpu.deinit(task)
