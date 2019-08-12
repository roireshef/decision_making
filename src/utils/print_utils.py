class PrintUtils:
    """ sets the colors of the print().
    See https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-terminal-in-python"""

    @staticmethod
    def change_bg_default():
        print('\x1b[0m')

    @staticmethod
    def change_bg_red():
        print('\x1b[6;30;41m')

    @staticmethod
    def change_bg_green():
        print('\x1b[6;30;42m')

    @staticmethod
    def change_bg_yellow():
        print('\x1b[5;30;43m')

    @staticmethod
    def change_bg_blue():
        print('\x1b[6;30;44m')

    @staticmethod
    def change_bg_purple():
        print('\x1b[6;30;45m')

