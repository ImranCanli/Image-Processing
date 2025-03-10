from keras.models import model_from_json
import numpy as np
from PIL import Image
import keyboard
import time
from mss import mss


mon = {"top": 370, "left": 770, "width": 250, "height": 90}
sct = mss()

width = 125
height = 50

# model yukle
model = model_from_json(open("model_new.json", "r").read())
model.load_weights("trex_weights_new.weights.h5")

# etiketler (down = 0, right = 1, up = 2)
labels = ["down", "right", "up"]


framerate_time = time.time()
counter = 0
i = 0
delay = 0.4

key_down_pressed = False

while True:
    
    img = sct.grab(mon)
    im = Image.frombytes("RGB", img.size, img.rgb)
    
    im2 =  np.array(im.convert("L").resize((width, height))) 
    im2 = im2/255
    
    X = np.array([im2])
    X = X.reshape(X.shape[0], width, height)
    r = model.predict(X)
    
    result = np.argmax(r)
    
    if result == 0: #down arow key must be pressed
        keyboard.press(keyboard.KEY_DOWN)
        key_down_pressed = True
    elif result == 2: #up arow key must be pressed
        
        if key_down_pressed == True:
            keyboard.release(keyboard.KEY_DOWN)
            time.sleep(delay)
            
    
        keyboard.press(keyboard.KEY_UP)
        
        if i < 1500:
            time.sleep(0.4) # trex oyunu 1500'uncu frame'e kadar belirli bir hızda ilerliyor. Bu yüzden dinozor zıpladıgında geri asagi inme komutunun islenmesi icin beklenmesi gereken sure, 0.3s olarak ayarlandı
        elif 1500 < i < 5000:
            time.sleep(0.2) # trex oyunu 1500'uncu ile 5000 frame arasında. Bu yuzden bekleme suresi, 0.2s olarak ayarlandı.
        else:
            time.sleep(0.15) # trex oyunu 5000'inci frame'den sonra baya hızlanıyor. Bu yuzden bekleme suresi, 0.17s olarak ayarlandı.
            
        keyboard.press(keyboard.KEY_DOWN)
        keyboard.release(keyboard.KEY_DOWN)
        
    counter += 1
    
    if (time.time() - framerate_time) > 1:
        counter = 0
        framerate_time = time.time() 
        
        if i <= 1500:
            delay -= 0.003
        else:
            delay -=0.005
        
        if delay < 0:
            delay = 0
            
        print("-------------------")
        print("Down: {} \nRight: {} \nUp: {} \n".format(r[0][0], r[0][1], r[0][2]))
        
        i = i + 1
        
            
        
        
        
        
        