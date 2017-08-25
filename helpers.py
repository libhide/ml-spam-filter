import os


def get_email_list(dir):
    filenames = [os.path.join(dir, f) for f in os.listdir(dir)]
    email_text_list = []
    for fil in filenames:
        with open(fil) as f:
            for i, line in enumerate(f):
                if i == 2:
                    email_text_list.append(line)
    return email_text_list
