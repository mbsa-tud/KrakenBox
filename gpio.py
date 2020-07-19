import time
import Jetson.GPIO as GPIO
#import RPi.GPIO as GPIO
#LEDPins = [13, 15, 18]
#LEDPins = [29, 37, 33]
buzzerPin = 7

def setup():
    GPIO.setmode(GPIO.BOARD)
    
    GPIO.setup(buzzerPin, GPIO.OUT)
    
    global pwmRed,pwmGreen,pwmBlue
    
#    GPIO.setup(LEDPins, GPIO.OUT)
#    GPIO.output(LEDPins, GPIO.HIGH)
    
#    pwmRed = GPIO.PWM(LEDPins[0], 2000)
#    pwmGreen = GPIO.PWM(LEDPins[1], 2000)
#    pwmBlue = GPIO.PWM(LEDPins[2], 2000)
#    pwmRed.start(0)
#    pwmGreen.start(0)
#    pwmBlue.start(0)

def destroy():
    GPIO.output(buzzerPin, GPIO.LOW)
#    pwmRed.stop()
#    pwmGreen.stop()
#    pwmBlue.stop()
    GPIO.cleanup()
    
    
    
setup() 

#start, green
#pwmRed.ChangeDutyCycle(100)
#pwmGreen.ChangeDutyCycle(100)
#pwmBlue.ChangeDutyCycle(0)

#time.sleep(5)

#residual>threshold, buzzer-high
#pwmRed.ChangeDutyCycle(0)
#pwmGreen.ChangeDutyCycle(100)
#pwmBlue.ChangeDutyCycle(100)
GPIO.output(buzzerPin,GPIO.HIGH)

time.sleep(5)

#residual<threshold, buzzer-low
#pwmRed.ChangeDutyCycle(100)
#pwmGreen.ChangeDutyCycle(100)
#pwmBlue.ChangeDutyCycle(0)
GPIO.output(buzzerPin,GPIO.LOW)

time.sleep(5)

destroy()




# =============================================================================
# #%%
# 
# import RPi.GPIO as GPIO
# import time
# 
# output_pins = {
#     'JETSON_XAVIER': 18,
#     'JETSON_NANO': 33,
#     'JETSON_NX': 33,
# }
# output_pin = output_pins.get(GPIO.model, None)
# if output_pin is None:
#     raise Exception('PWM not supported on this board')
# 
# 
# def main():
#     # Pin Setup:
#     # Board pin-numbering scheme
#     GPIO.setmode(GPIO.BOARD)
#     # set pin as an output pin with optional initial state of HIGH
#     GPIO.setup(output_pin, GPIO.OUT, initial=GPIO.HIGH)
#     p = GPIO.PWM(output_pin, 50)
#     val = 25
#     incr = 5
#     p.start(val)
# 
#     print("PWM running. Press CTRL+C to exit.")
#     try:
#         while True:
#             time.sleep(0.25)
#             if val >= 100:
#                 incr = -incr
#             if val <= 0:
#                 incr = -incr
#             val += incr
#             p.ChangeDutyCycle(val)
#     finally:
#         p.stop()
#         GPIO.cleanup()
# 
# if __name__ == '__main__':
#     main()
# =============================================================================
