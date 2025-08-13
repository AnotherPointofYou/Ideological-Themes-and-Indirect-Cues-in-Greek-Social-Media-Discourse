import re, requests, os, string
import concurrent.futures
from typing import List
import spacy
from greek_stemmer import stemmer
from dotenv import load_dotenv
from gr_nlp_toolkit import Pipeline
from tqdm import tqdm

class data_cleaning:

    SPACY_TO_ELLOGON_POS = {
        "NOUN": "NNM", # noun masc
        "PROPN": "NNPM", # proper noun masc
        "VERB": "VB", # verb base
        "AUX":  "VB", # auxiliary as base
        "ADJ":  "JJM", # adjective masc
        "ADV":  "RB", # adverb
        "DET":  "DDT", # determiner
        "ADP":  "INP", # adposition (prep)
        "PRON": "PRP", # pronoun
        "NUM":  "CD", # numeral
    }

    g2g: Pipeline

    # Greek spaCy model for POS tagging
    nlp = spacy.load("el_core_news_sm")

    def __init__(self):
        """
        Initialize spacy's Greek corpus and, Greeklish to Greek and POS tagging pipelines
        from Greek NLP toolkit.
        """
        loaders = [
            ("g2g", lambda: Pipeline("g2g")),
            ("pos", lambda: Pipeline("pos")),
            ("nlp", lambda: spacy.load("el_core_news_sm"))
        ]
        for name, loader in tqdm(loaders,
                                 desc="Initializing components",
                                 unit="component"):
            setattr(self, name, loader())

        # self.g2g = Pipeline("g2g")
        # self.pos = Pipeline("pos")
        # self.nlp = spacy.load("el_core_news_sm")
        self.stemmer = stemmer

        load_dotenv()
        self.DEEPL_API_KEY = os.getenv("DEEPL_API_KEY") # find the key from the .env file
        self.DEEPL_URL = os.getenv("DEEPL_URL") # find the password from the .env file

        # validate environment variables
        if not self.DEEPL_API_KEY or not self.DEEPL_URL:
            raise ValueError("DeepL API key or URL is missing in environment variables.")

    @staticmethod
    def remove_greek_accents(text: str) -> str:
        text = text.translate(str.maketrans('άέόώήύϋΰίϊΐ', 'αεοωηυυυιιι'))
        return text

    @staticmethod
    def normalize(text: str) -> str:
        """
        1) Remove Giphy‐style embeds: ![gif](giphy|...|...)
        2) Remove all other Markdown images: ![alt](...)
        3) Drop any parenthesized URL: (http://…)
        4) Drop bare URLs (http/https/www)
        5) Remove HTML tags: <...>
        6) Remove all emojis
        7) Remove digits
        8) remove Greek accents
        9) remove special characters (keep letters, spaces)
        10) Collapse whitespace, lowercase
        """
        # strip Giphy embeds
        text = re.sub(r'!\[[^\]]*\]\(giphy\|[^)]+\)', ' ', text)
        # strip other Markdown images
        text = re.sub(r'!\[[^\]]*\]\([^\)]*\)', ' ', text)
        # strip parenthesized URLs
        text = re.sub(r'\(\s*https?://[^)]+\)', ' ', text)
        # strip bare URLs
        text = re.sub(r'http\S+|www\.\S+|https\S+', ' ', text)
        # strip HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # remove emojis
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F" # emoticons
            "\U0001F300-\U0001F5FF" # symbols & pictographs
            "\U0001F680-\U0001F6FF" # transport & map
            "\U0001F1E0-\U0001F1FF" # flags
            "\U00002702-\U000027B0" # dingbats
            "\U000024C2-\U0001F251" # enclosed chars
            "\U0000200D" # zero-width joiner
            "\U0001F900-\U0001F9FF" # supplemental symbols
            "\U0001FA70-\U0001FAFF" # extended-A
            "\U0001FAD0-\U0001FADF" # food & drink
            "]+", flags=re.UNICODE
        )

        text = emoji_pattern.sub(' ', text)
        # remove digits
        text = re.sub(r'\d+', ' ', text)
        # remove special characters (keep letters, spaces)
        text = re.sub(r'[^\w\s@.?!;]', ' ', text, flags=re.UNICODE)
        text = text.replace('_', ' ')
        # collapse whitespace & lowercase
        text = re.sub(r'\s+', ' ', text).strip().lower()
        # remove Greek accents
        text = data_cleaning.remove_greek_accents(text)
        return text

    @staticmethod
    def contains_mixed_latin_greek(text: str) -> str:
        """
        Return True if `text` contains at least one Greek-letter character
        and at least one Latin-letter character.
        """
        _GREEK_PATTERN = re.compile(r'[\u0370-\u03FF\u1F00-\u1FFF]')
        _LATIN_PATTERN = re.compile(r'[A-Za-z]')

        has_greek = bool(_GREEK_PATTERN.search(text))
        has_latin = bool(_LATIN_PATTERN.search(text))

        if has_greek and not has_latin:
            return "Greek"
        elif has_latin and not has_greek:
            return "Latin"
        elif has_greek and has_latin:
            return "Mixed"
        else:
            return None

    @staticmethod
    def word_count(text: str) -> int:
        """
        It counts the words in a given text

        Args:
            text (str): Given text
        
        Returns:
            len(words) (int): Word count
        """
        # clean text
        clean_text = data_cleaning.normalize(text)
        # replace punctuation with spaces
        new_text = clean_text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        # split text
        words = new_text.split()
        return len(words) # count its length

    def translate_to_greek(self, text):
        """
        Translates the input text to English using the DeepL API.

        Args:
            text (str): The text to be translated.

        Returns:
            str or None: The translated text if successful, or None if an error occurs.
        """
        try:
            params = {
                "auth_key": self.DEEPL_API_KEY, # use the API key
                "text": text, # text
                "target_lang": "EL" # translation to Greek language
            }

            response = requests.post(self.DEEPL_URL, data=params, timeout=10) # API response (translation)

            if response.status_code == 200: # if the response is successful acquire the translated text
                translation = response.json()["translations"][0]
                source_language = translation.get("detected_source_language", "").upper()

                if source_language != "EN": # if its not english and it is another language we assume it is either wrong or Greeklish
                    return "NE"

                return translation["text"]
            else:
                print("Error:", response.status_code, response.text)
                return None

        except requests.exceptions.RequestException as error: # raise exception regarding the request
            print("Request Error:", error)
            return None
        except Exception as error: # raise exception regarding unknown reason
            print("Error:", error)
            return None

    def keep_only_greek(self, text: str) -> str:
        """
        Remove any character that isn’t in the Greek Unicode blocks
        (U+0370–03FF, U+1F00–1FFF), replace runs with a single space.
        """
        cleaned = re.sub(r"[^ \u0370-\u03FF\u1F00-\u1FFF]+", " ", text)
        return cleaned.strip()

    def safe_g2g(self, token, timeout=10):
        def task():
            return self.g2g(token).text

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(task)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                print(f"[Token Timeout] Skipping token: {token}")
                return token  # fallback: return unchanged
            except Exception as e:
                print(f"[G2G Error] {e} on token: {token}")
                return token

    def transliterate(self, text: str) -> str:
        """
        ex-convert_greeklish
        Only transliterate if G2G actually changes the text.
        Pure-English or pure-Greek is returned unchanged.
        """
        if self.contains_mixed_latin_greek(text) == "Latin":
            transl_txt = self.translate_to_greek(text)
            print(transl_txt)
            if transl_txt == "NE" or None: # probably Greeklish
                # converted = self.safe_g2g(text)  # ← safe per-token call
                return text
            else:
                return transl_txt
        else:
            return text

    def stem(self, text: str) -> str:
        """
        POS‐aware stemming via Ellogon:
        1. run SpaCy to get token.pos_
        2. map to Ellogon POS codes
        3. stem_word(token.upper(), pos_code)
        """
        doc = self.nlp(text)
        out = []
        for token in doc:
            pos_code = self.SPACY_TO_ELLOGON_POS.get(token.pos_, "NNM")
            try:
                s = self.stemmer.stem_word(token.text.upper(), pos_code)
                out.append(s)
            except ValueError:
                # skip tokens that Ellogon cannot stem
                continue
        return " ".join(out)

    def remove_greek_stopwords(self, text: str) -> str:
        """
        Removes Greek stopwords from a space-tokenized string.

        Args:
            text (str): A cleaned string of space-separated words.
            stopwords (set): A set of stopwords (no accents).

        Returns:
            str: The text without stopwords.
        """
        greek_stopwords = set(
            data_cleaning.remove_greek_accents("""
            αδιάκοπα αι ακόμα ακόμη ακριβώς άλλα αλλά αλλαχού άλλες άλλη άλλην
            άλλης αλλιώς αλλιώτικα άλλο άλλοι αλλοιώς αλλοιώτικα άλλον άλλος άλλοτε αλλού
            άλλους άλλων άμα άμεσα αμέσως αν ανά ανάμεσα αναμεταξύ άνευ αντί αντίπερα αντίς
            άνω ανωτέρω άξαφνα απ απέναντι από απόψε άρα άραγε αρκετά αρκετές
            αρχικά ας αύριο αυτά αυτές αυτή αυτήν αυτής αυτό αυτοί αυτόν αυτός αυτού αυτούς
            αυτών αφότου αφού

            βέβαια βεβαιότατα

            γι για γιατί γρήγορα γύρω

            δα δε δείνα δεν δεξιά δήθεν δηλαδή δι δια διαρκώς δικά δικό δικοί δικός δικού
            δικούς διόλου δίπλα δίχως

            εάν εαυτό εαυτόν εαυτού εαυτούς εαυτών έγκαιρα εγκαίρως εγώ εδώ ειδεμή είθε είμαι
            είμαστε είναι εις είσαι είσαστε είστε είτε είχα είχαμε είχαν είχατε είχε είχες έκαστα
            έκαστες έκαστη έκαστην έκαστης έκαστο έκαστοι έκαστον έκαστος εκάστου εκάστους εκάστων
            εκεί εκείνα εκείνες εκείνη εκείνην εκείνης εκείνο εκείνοι εκείνον εκείνος εκείνου
            εκείνους εκείνων εκτός εμάς εμείς εμένα εμπρός εν ένα έναν ένας ενός εντελώς εντός
            εναντίον  εξής  εξαιτίας  επιπλέον επόμενη εντωμεταξύ ενώ εξ έξαφνα εξήσ εξίσου έξω επάνω
            επειδή έπειτα επί επίσης επομένως εσάς εσείς εσένα έστω εσύ ετέρα ετέραι ετέρας έτερες
            έτερη έτερης έτερο έτεροι έτερον έτερος ετέρου έτερους ετέρων ετούτα ετούτες ετούτη ετούτην
            ετούτης ετούτο ετούτοι ετούτον ετούτος ετούτου ετούτους ετούτων έτσι εύγε ευθύς ευτυχώς εφεξής
            έχει έχεις έχετε έχομε έχουμε έχουν εχτές έχω έως έγιναν  έγινε  έκανε  έξι  έχοντας

            η ήδη ήμασταν ήμαστε ήμουν ήσασταν ήσαστε ήσουν ήταν ήτανε ήτοι ήττον

            θα

            ι ιδία ίδια ίδιαν ιδίας ίδιες ίδιο ίδιοι ίδιον ίδιοσ ίδιος ιδίου ίδιους ίδιων ιδίως ιι ιιι
            ίσαμε ίσια ίσως

            κάθε καθεμία καθεμίας καθένα καθένας καθενός καθετί καθόλου καθώς και κακά κακώς καλά
            καλώς καμία καμίαν καμίας κάμποσα κάμποσες κάμποση κάμποσην κάμποσης κάμποσο κάμποσοι
            κάμποσον κάμποσος κάμποσου κάμποσους κάμποσων κανείς κάνεν κανένα κανέναν κανένας
            κανενός κάποια κάποιαν κάποιας κάποιες κάποιο κάποιοι κάποιον κάποιος κάποιου κάποιους
            κάποιων κάποτε κάπου κάπως κατ κατά κάτι κατιτί κατόπιν κάτω κιόλας κλπ κοντά κτλ κυρίως

            λιγάκι λίγο λιγότερο λόγω λοιπά λοιπόν

            μα μαζί μακάρι μακρυά μάλιστα μάλλον μας με μεθαύριο μείον μέλει μέλλεται μεμιάς μεν
            μερικά μερικές μερικοί μερικούς μερικών μέσα μετ μετά μεταξύ μέχρι μη μήδε μην μήπως
            μήτε μια μιαν μιας μόλις μολονότι μονάχα μόνες μόνη μόνην μόνης μόνο μόνοι μονομιάς
            μόνος μόνου μόνους μόνων μου μπορεί μπορούν μπρος μέσω  μία  μεσώ

            να ναι νωρίς

            ξανά ξαφνικά

            ο οι όλα όλες όλη όλην όλης όλο ολόγυρα όλοι όλον ολονέν όλος ολότελα όλου όλους όλων
            όλως ολωσδιόλου όμως όποια οποιαδήποτε οποίαν οποιανδήποτε οποίας οποίος οποιασδήποτε οποιδήποτε
            όποιες οποιεσδήποτε όποιο οποιοδηήποτε όποιοι όποιον οποιονδήποτε όποιος οποιοσδήποτε
            οποίου οποιουδήποτε οποίους οποιουσδήποτε οποίων οποιωνδήποτε όποτε οποτεδήποτε όπου
            οπουδήποτε όπως ορισμένα ορισμένες ορισμένων ορισμένως όσα οσαδήποτε όσες οσεσδήποτε
            όση οσηδήποτε όσην οσηνδήποτε όσης οσησδήποτε όσο οσοδήποτε όσοι οσοιδήποτε όσον οσονδήποτε
            όσος οσοσδήποτε όσου οσουδήποτε όσους οσουσδήποτε όσων οσωνδήποτε όταν ότι οτιδήποτε
            ότου ου ουδέ ούτε όχι οποία  οποίες  οποίο  οποίοι  οπότε  ος

            πάνω  παρά  περί  πολλά  πολλές  πολλοί  πολλούς  που  πρώτα  πρώτες  πρώτη  πρώτο  πρώτος  πως
            πάλι πάντα πάντοτε παντού πάντως πάρα πέρα πέρι περίπου περισσότερο πέρσι πέρυσι πια πιθανόν
            πιο πίσω πλάι πλέον πλην ποιά ποιάν ποιάς ποιές ποιό ποιοί ποιόν ποιός ποιού ποιούς
            ποιών πολύ πόσες πόση πόσην πόσης πόσοι πόσος πόσους πότε ποτέ πού πούθε πουθενά πρέπει
            πριν προ προκειμένου πρόκειται πρόπερσι προς προτού προχθές προχτές πρωτύτερα πώς

            σαν σας σε σεις σου στα στη στην στης στις στο στον στου στους στων συγχρόνως
            συν συνάμα συνεπώς συχνάς συχνές συχνή συχνήν συχνής συχνό συχνοί συχνόν
            συχνός συχνού συχνούς συχνών συχνώς σχεδόν

            τα τάδε ταύτα ταύτες ταύτη ταύτην ταύτης ταύτοταύτον ταύτος ταύτου ταύτων τάχα τάχατε
            τελευταία  τελευταίο  τελευταίος  τού  τρία  τρίτη  τρεις τελικά τελικώς τες τέτοια τέτοιαν
            τέτοιας τέτοιες τέτοιο τέτοιοι τέτοιον τέτοιος τέτοιου
            τέτοιους τέτοιων τη την της τι τίποτα τίποτε τις το τοι τον τοσ τόσα τόσες τόση τόσην
            τόσης τόσο τόσοι τόσον τόσος τόσου τόσους τόσων τότε του τουλάχιστο τουλάχιστον τους τούς τούτα
            τούτες τούτη τούτην τούτης τούτο τούτοι τούτοις τούτον τούτος τούτου τούτους τούτων τυχόν
            των τώρα

            υπ υπέρ υπό υπόψη υπόψιν ύστερα

            χωρίς χωριστά

            ω ως ωσάν ωσότου ώσπου ώστε ωστόσο ωχ κ κι
            """).split()
        )
        return " ".join(word for word in text.split() if word not in greek_stopwords)

    def youtube_specific(self, text: str) -> str:
        """
        Any YouTube specific noise cleaning.
        E.g. @account_name in replies remains
        """
        text = re.sub(r"^@\S+\s+", "", text) # for remaining replying names
        direct_phrases = ["quot", "href"]
        for phrase in direct_phrases:
            text = text.replace(phrase, "")

        text = re.sub(r'@\w+', ' ', text)
        cleaned = re.sub(r'\s+', ' ', text).strip()

        return cleaned

    def reddit_specific(self, text: str) -> str:
        """
        Any Reddit specific noise cleaning.
        E.g. [deleted] when comment is deleted from the user or moderator
        """
        direct_phrases = ["deleted", "removed", "quot", "href"]
        full_removal_triggers = [
            "δεν επιτρέπονται σύνδεσμοι προς σελίδες google amp το σχόλιό σου έχει αφαιρεθεί μπορείς όμως να το",
            "δεν επιτρεπονται συνδεσμοι προς σελιδες google amp το σχολιο σου εχει αφαιρεθει μπορεις ομως να το επεξεργαστεις και να ενημερωσεις τους",
            "ο τίτλος στο σαιτ αλλάζει συνεχώς για αυτό πιθανότατα",
            "ο τιτλος στο σαιτ αλλαζει συνεχως για αυτο πιθανοτατα"
        ]

        for trigger in full_removal_triggers:
            if trigger in text:
                return ""

        for phrase in direct_phrases:
            text = text.replace(phrase, "")

        text = re.sub(r"\s+", " ", text).strip()
        return text

class filtering_pipelines(data_cleaning):

    def __init__(self):
        super().__init__()

    def filter_content(self, sentence: str, phrases: List[str]) -> bool:
        """
        Accepts keywords and by applying filtering steps sequentially it returns
        a boolean output for whether the text stems matche with the keywords' stems.
        """
        # sentence processing
        sent_norm = self.normalize(sentence)
        sent_conv = self.transliterate(sent_norm)
        sent_stem = self.stem(sent_conv)
        # check each phrase
        for ph in phrases:
            p_norm = self.normalize(ph)
            p_conv = self.transliterate(p_norm)
            p_stem = self.stem(p_conv)
            match = p_stem in sent_stem
            if match:
                return True
        return False

class cleaning_pipelines(data_cleaning):

    def __init__(self):
        super().__init__()

    def text_cleaning(self, text: str, steps: list[str]) -> str:
        """
        Sequentially apply each named method in "steps" to "text".
            steps: all data_cleaning steps.
        """
        for step in steps:
            print(f"Step {step} started !!")
            if not hasattr(self, step):
                raise ValueError(f"Step '{step}' not found in data_cleaning_pipeline")
            method = getattr(self, step)
            text = method(text)
        return text
