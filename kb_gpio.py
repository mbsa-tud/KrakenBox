import time
import Jetson.GPIO as GPIO

# xavier bright pin 8, 11
ledPin = {'r':33,'g':29,'y':37,'buzzer':7} #led pin of the black-extension box

def setup():
    #start, green
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup([33,29,37,7], GPIO.OUT)
    
    GPIO.output(ledPin['r'], GPIO.LOW)
    GPIO.output(ledPin['g'], GPIO.HIGH)
    GPIO.output(ledPin['y'],GPIO.LOW)
    GPIO.output(ledPin['buzzer'], GPIO.LOW)
    
#def detecting(flag):
#    if flag==1:#residual>threshold,
#        GPIO.output(ledPin['r'], GPIO.HIGH)
#        GPIO.output(ledPin['g'], GPIO.LOW)
#        GPIO.output(ledPin['buzzer'], GPIO.HIGH)
#    else:
#        GPIO.output(ledPin['r'], GPIO.LOW)
#        GPIO.output(ledPin['g'], GPIO.HIGH)
#        GPIO.output(ledPin['buzzer'], GPIO.LOW)

        
    
def red_alarm():
    GPIO.output(ledPin['r'],GPIO.HIGH)
    GPIO.output(ledPin['y'],GPIO.LOW)
    GPIO.output(ledPin['g'],GPIO.LOW)
    GPIO.output(ledPin['buzzer'], GPIO.HIGH)
    
    
def green_normal():
    GPIO.output(ledPin['g'],GPIO.HIGH)
    GPIO.output(ledPin['y'],GPIO.LOW)
    GPIO.output(ledPin['r'],GPIO.LOW)
    GPIO.output(ledPin['buzzer'], GPIO.LOW)
    

def yellow_training():
    GPIO.output(ledPin['y'],GPIO.HIGH)
    GPIO.output(ledPin['r'],GPIO.LOW)
    GPIO.output(ledPin['g'],GPIO.LOW)
    GPIO.output(ledPin['buzzer'], GPIO.LOW)

        
def testing():
    while True:
        green_normal()
        time.sleep(5)
        red_alarm()
        time.sleep(0.1)
        
        
def destroy():
    GPIO.output(ledPin['r'], GPIO.LOW)
    GPIO.output(ledPin['g'], GPIO.LOW)
    GPIO.output(ledPin['y'], GPIO.LOW)
    GPIO.output(ledPin['buzzer'], GPIO.LOW)
    GPIO.cleanup()
    
    
if __name__ == '__main__':
    setup()
    print(__name__)
    
    try:
        testing()
    except KeyboardInterrupt:
        destroy()





    
 


