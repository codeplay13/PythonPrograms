import speech_recognition as sr
import pyttsx3

speak = pyttsx3.init()
speak.say('Say something fella')

r = sr.Recognizer()
with sr.Microphone() as source:
	audio = r.listen(source)
try:
	print('You said: ' + r.recognize_google(audio))
	#speak.say('Did you say' + r.recognize_google(audio))
except:
	print('Couldn\'t hear ya!')
speak.runAndWait()