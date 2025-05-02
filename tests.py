import nltk
nltk.data.path.append("E:/Tester/ESN Modelling/esn_scraping/nltk_data")

try:
    from nltk.tokenize import sent_tokenize
    sent_tokenize("Test sentence.")  # test run
except LookupError:
    nltk.download('punkt', download_dir="E:/Tester/ESN Modelling/esn_scraping/nltk_data")
