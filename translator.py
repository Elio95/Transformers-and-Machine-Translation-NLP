from google_trans_new import google_translator
import csv
import os

translator = google_translator()


def translate_csv(dir, file, target_lang):
    csv_temp = []
    with open(dir+'/'+file, "r", encoding="UTF-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        csv_temp.append(header)
        for line in reader:
            translation = translator.translate(line[1].encode('unicode_escape'), lang_tgt='en')
            csv_temp.append([line[0], translation])
    with open(dir+'/' + target_lang+file, 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for line in csv_temp:
            tsv_writer.writerow(line)


dataset_dir = os.path.join(".", "datasets/chnsenticorp")
translate_csv(dataset_dir, 'test.tsv', 'en')
