import sensor,image,lcd,time
import KPU as kpu

lcd.init(freq=15000000)
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_hmirror(0)
#sensor.set_vflip(1)
sensor.set_windowing((224, 224))
sensor.set_brightness(2)
#sensor.set_contrast(-1)
#sensor.set_auto_gain(1,2)

sensor.run(1)
clock = time.clock()
#classes = ['black+black+red', 'black+black+silver', 'black+white+red', 'black+white+silver', 'white+black+red', 'white+black+silver', 'white+white+red', 'white+white+silver']
classes = ['bbr','bbs','bwr','bws','wbr','wbs','wwr','wws']
task = kpu.load(0x300000)
anchor = [1.51,2.43, 1.83,1.54, 2.26,3.08, 2.59,2.07, 3.75,3.41]
a = kpu.init_yolo2(task, 0.5, 0.3, 5, anchor)
while(True):
    clock.tick()
    img = sensor.snapshot()
    code = kpu.run_yolo2(task, img)
    #print(clock.fps())
    print(code)
    if code:
        for i in code:
            a=img.draw_rectangle(i.rect())
            print(i.classid(),i.value())
            for i in code:
                lcd.draw_string(i.x(), i.y(), classes[i.classid()], lcd.RED, lcd.WHITE)
                lcd.draw_string(i.x(), i.y()+12, '%f1.3'%i.value(), lcd.RED, lcd.WHITE)
            a = lcd.display(img)

    else:
        a = lcd.display(img)
a = kpu.deinit(task)



