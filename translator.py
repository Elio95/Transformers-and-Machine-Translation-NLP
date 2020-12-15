from google_trans_new import google_translator
import csv
import os
import codecs

translator = google_translator()


def translate_csv(dir, file, target_lang):
    csv_temp = []
    with open(os.path.join(dir, file), "rt", encoding="UTF-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        csv_temp.append(header)
        for line in reader:
            translation = translator.translate(line[1], lang_tgt=target_lang)
            csv_temp.append([line[0], translation])
            print(len(csv_temp))
    with open(dir+'/' + target_lang+"_"+file, 'w') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for line in csv_temp:
            tsv_writer.writerow(line)

def split_imdb():
    dir = os.path.join(".", "datasets/IMDB")

    with open(os.path.join(dir, "rt-polarity.neg"), 'rb') as neg,\
            open(os.path.join(dir, "rt-polarity.pos"), 'rb') as pos,\
            open(os.path.join(dir, "test.tsv"), 'wt') as test, \
            open(os.path.join(dir, "train.tsv"), 'wt') as train, \
            open(os.path.join(dir, "dev.tsv"), 'wt') as dev:

        lines = neg.readlines()
        num_train = int(len(lines)*0.8)
        num_test = num_train + int(len(lines)*0.10)

        test_writer = csv.writer(test, delimiter='\t')
        train_writer = csv.writer(train, delimiter='\t')
        dev_writer = csv.writer(dev, delimiter='\t')
        for i, line in enumerate(lines):
            if 0<= i <num_train:
                train_writer.writerow([0, line.decode('unicode_escape').strip()])
            elif i<num_test:
                test_writer.writerow([0, line.decode('unicode_escape').strip()])
            else:
                dev_writer.writerow([0, line.decode('unicode_escape').strip()])

        lines = pos.readlines()
        num_train = int(len(lines)*0.8)
        num_test = num_train + int(len(lines)*0.10)

        test_writer = csv.writer(test, delimiter='\t')
        train_writer = csv.writer(train, delimiter='\t')
        dev_writer = csv.writer(dev, delimiter='\t')
        for i, line in enumerate(lines):
            if 0<= i <num_train:
                train_writer.writerow([1, line.decode('unicode_escape').strip()])
            elif i<num_test:
                test_writer.writerow([1, line.decode('unicode_escape').strip()])
            else:
                dev_writer.writerow([1, line.decode('unicode_escape').strip()])


dir = os.path.join(".", "datasets/chnsenticorp")
translate_csv(dir, "dev.tsv", "en")

dir = os.path.join(".", "datasets/IMDB")
translate_csv(dir, "dev.tsv", "zh")
translate_csv(dir, "test.tsv", "zh")
translate_csv(dir, "train.tsv", "zh")

