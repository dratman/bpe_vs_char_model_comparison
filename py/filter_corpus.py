"""
filter_corpus.py — Classify Gutenberg texts as keep/remove for corpus cleanup.

Reads gutenberg_data.csv and the gutenberg_texts/ directory to classify
each file. Uses Gutenberg bookshelf categories where available, plus
filename pattern matching for the ~6800 uncategorized files.

Criteria:
  KEEP: novels, short stories, literary essays, memoirs, travel writing,
        biography, literary criticism, philosophy in literary English,
        slave narratives, children's fiction, adventure, detective fiction
  REMOVE: architecture manuals, religious scriptures, math constants,
          CIA factbooks, cookbooks, engineering texts, non-English works,
          encyclopedias, dictionaries, zoology taxonomy papers, poetry,
          grammars, primers, medical treatises, military manuals,
          periodicals, very short fragments (<3KB)

Output: two files in the working directory:
  corpus_keep.txt    — filenames to keep (one per line)
  corpus_remove.txt  — filenames to remove with reason (tab-separated)
"""

import csv
import os
import re

GUTENBERG_DIR = '/Users/RalphDratman/Library/Mobile Documents/com~apple~CloudDocs/0-HomeFolder-Working-iCloud_A/Gutenberg_Project_Books'
DATA_CSV = os.path.join(GUTENBERG_DIR, 'gutenberg_data.csv')
TEXTS_DIR = os.path.join(GUTENBERG_DIR, 'gutenberg_texts')

# Bookshelves to REMOVE entirely
REMOVE_SHELVES = {
    'Architecture', 'Archaeology', 'Astronomy', 'Atheism',
    'Bahá\'í', 'Biology', 'Botany', 'Buddhism',
    'Chemistry', 'CIA', 'Cookbooks', 'Crafts',
    'Czech', 'Ecology', 'Education', 'Engineering',
    'FR', 'German', 'Geology',
    'Hinduism', 'Islam', 'IT', 'Judaism',
    'Language', 'Manufacturing', 'Mathematics', 'Microbiology',
    'Microscopy', 'Music', 'Mycology', 'Opera',
    'Paganism', 'Physics', 'Physiology', 'Psychology',
    'Reference', 'School', 'Scientific', 'Sociology',
    'Spanish', 'Technology', 'Transportation', 'Woodwork',
    'Zoology',
}

# Bookshelves to KEEP entirely
KEEP_SHELVES = {
    'Adventure', 'Animal',  # some animal books are literary (Jack London etc)
    'Biographies', 'British', 'Camping',
    'Canada', 'Children\'s', 'Child\'s', 'Christmas',
    'Crime', 'Current', 'Detective',
    'Egypt', 'Erotic', 'Fantasy', 'Folklore',
    'France', 'Gothic', 'Greece', 'Historical',
    'Horror', 'Humor', 'India', 'Italy',
    'Latter',  # review these individually
    'Mexico', 'Mystery', 'Mythology',
    'Natural', 'Noteworthy', 'Philosophy',
    'Precursors', 'Romantic', 'Slavery', 'Suffrage',
    'Travel', 'Western', 'Witchcraft', 'Women\'s',
    'Africa', 'Argentina', 'Australia', 'Germany',
    'South', 'United',  # US history etc
    'US', 'New',
    'Anarchism',  # political philosophy, literary
    'Art',  # mixed, but many are literary criticism
    'Bibliomania',  # book collecting, literary
    'Canon',
    'Classical',  # classical literature
}

# Filename patterns that indicate REMOVAL (case-insensitive)
REMOVE_PATTERNS = [
    # Non-English
    (r'(?:^|\b)(?:Le_|La_|Les_|Un_|Une_|Des_|Du_|Au_|Aux_)', 'non-English (French)'),
    (r'Memoires_d', 'non-English (French)'),
    (r'Souvenirs_d', 'non-English (French)'),
    (r'Journal_des_Goncourt', 'non-English (French)'),
    (r'Correspondance_', 'non-English (French)'),
    (r'(?:^|\b)(?:Der_|Die_|Das_|Ein_|Eine_|Vom_|Wie_)', 'non-English (German)'),
    (r'(?:^|\b)(?:Het_|Een_|Naar_|Uit_|Van_|Door_)', 'non-English (Dutch)'),
    (r'(?:^|\b)(?:Il_|Gli_|Della_|Delle_|Ricordi_)', 'non-English (Italian)'),
    (r'Qvo_vadis_Kertomus', 'non-English (Finnish)'),
    (r'Kullankaivajat', 'non-English (Finnish)'),
    (r'_la_cour_', 'non-English (French)'),

    # Religious scriptures
    (r'(?:The_)?(?:Koran|Quran|Al-Quran)', 'religious scripture'),
    (r'(?:The_)?Bible.*(?:King_James|KJV|Reina_Valera)', 'religious scripture'),
    (r'Book_of_Mormon', 'religious scripture'),
    (r'Doctrine_and_Covenants', 'religious scripture'),
    (r'Pearl_of_Great_Price', 'religious scripture'),
    (r'Summa_Theologica', 'religious scripture'),
    (r'Latin_Vulgate', 'religious scripture'),
    (r'Expositions_of_Holy_Scripture', 'religious text'),
    (r'(?:Det_Gamle|Nye)_Testamente', 'non-English Bible'),
    (r'Bibeln_Gamla', 'non-English Bible'),

    # CIA World Factbook
    (r'CIA_World_Factbook', 'CIA factbook'),

    # Mathematics / numerical data
    (r'Square_Root_of', 'math data'),
    (r'Fibonacci_Num', 'math data'),
    (r'Euler_Numbers', 'math data'),
    (r'Bernoulli_Numbers', 'math data'),
    (r'Mersenne_Prime', 'math data'),
    (r'Miscellaneous_Mathematical', 'math data'),
    (r'Catalans_Constant', 'math data'),
    (r'Golden_Mean', 'math data'),
    (r'Factorial_Math', 'math data'),
    (r'Value_of_Zeta', 'math data'),
    (r'Number_e\b', 'math data'),
    (r'One_Divided', 'math data'),
    (r'^[0-9]+_Pi\b', 'math data'),

    # Encyclopedias, dictionaries, thesauri, almanacs
    (r'Encyclop[ae]dia_Britannica', 'encyclopedia'),
    (r'Nuttall_Encyclop', 'encyclopedia'),
    (r'Gutenberg_Encyclopedia', 'encyclopedia'),
    (r'Rogets_Thesaurus', 'reference'),
    (r'Jargon_File', 'reference'),
    (r'Fifteen_Thousand_Useful_Phrases', 'reference'),
    (r'Dictionary_of_the_Vulgar', 'reference'),
    (r'Foolish_Dictionary', 'reference'),
    (r'Barkham_Burroughs', 'reference'),
    (r'Ten_Thousand_Dreams_Interpreted', 'reference'),

    # Engineering / technical
    (r'Transactions_of_the_American_Society', 'engineering'),
    (r'Scientific_American_Supplement', 'periodical'),
    (r'Catechism_of_the_Steam', 'engineering'),
    (r'Oxy-Acetylene', 'engineering'),
    (r'Nitro-Explosives', 'engineering'),
    (r'Cyclopedia_of_Telephony', 'engineering'),
    (r'Operators?_Manual', 'technical manual'),
    (r'TRS-80', 'technical manual'),
    (r'IBM_Programming', 'technical manual'),
    (r'Delco_Radio', 'technical manual'),
    (r'Marvel_Carbureter', 'technical manual'),
    (r'Motorcycle_Solo.*Harley', 'military manual'),
    (r'Portable_Flame_Thrower', 'military manual'),
    (r'Military_Instructors_Manual', 'military manual'),

    # Mirror of Literature periodical
    (r'Mirror_of_Literature.*Instruction', 'periodical'),

    # Nuremberg trials
    (r'Trial_of_the_Major_War_Criminals', 'trial transcript'),
    (r'Trials_of_War_Criminals', 'trial transcript'),

    # Sermons (bulk)
    (r'(?:Parochial_and_Plain_)?Sermons', 'sermons'),
    (r'Quiet_Talks_on', 'sermons'),
    (r'Village_Pulpit', 'sermons'),

    # Hymn books
    (r'Hymns?_(?:from|and|Songs)', 'hymn book'),
    (r'Otterbein_Hymnal', 'hymn book'),
    (r'Prayer_Book', 'religious reference'),

    # Cookbooks by pattern
    (r'(?:Recipes?|Cook_Book|Cookbook|Cooking)', 'cookbook'),
    (r'(?:365_)?Foreign_Dishes', 'cookbook'),

    # Constitution/legal (keep Declaration but remove annotated legal texts)
    (r'Constitution.*Annotated', 'legal reference'),
    (r'Amendments_to_the.*Constitution', 'legal reference'),
]

# Filename patterns that indicate KEEP (override removal)
KEEP_PATTERNS = [
    r'Picture_of_Dorian_Gray',
    r'Frankenstein',
    r'Dracula',
    r'(?:Sherlock|Holmes)',
    r'Jane_Eyre',
    r'Pride_and_Prejudice',
    r'Wuthering_Heights',
    r'Great_Expectations',
    r'Oliver_Twist',
    r'Tale_of_Two_Cities',
    r'Moby_Dick',
    r'Adventures_of_(?:Tom_Sawyer|Huckleberry)',
    r'War_and_Peace',
    r'Anna_Karenina',
    r'Count_of_Monte_Cristo',
]


def classify_by_shelf(shelf):
    """Return 'keep', 'remove', or None based on bookshelf."""
    if not shelf:
        return None
    # Check first word of shelf against our lists
    first_word = shelf.split()[0] if shelf else ''
    for s in REMOVE_SHELVES:
        if shelf.startswith(s) or first_word == s:
            return 'remove'
    for s in KEEP_SHELVES:
        if shelf.startswith(s) or first_word == s:
            return 'keep'
    return None


def classify_by_filename(filename):
    """Return ('remove', reason) or ('keep', None) based on filename patterns."""
    # Check keep patterns first (override)
    for pat in KEEP_PATTERNS:
        if re.search(pat, filename, re.IGNORECASE):
            return 'keep', None

    # Check remove patterns
    for pat, reason in REMOVE_PATTERNS:
        if re.search(pat, filename, re.IGNORECASE):
            return 'remove', reason

    return None, None


def main():
    # Load CSV data
    csv_data = {}
    with open(DATA_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fp = row.get('FilePath', '')
            basename = os.path.basename(fp)
            csv_data[basename] = row

    # Get all actual files
    all_files = sorted(os.listdir(TEXTS_DIR))

    keep = []
    remove = []  # (filename, reason)
    stats = {'keep': 0, 'remove': 0, 'keep_bytes': 0, 'remove_bytes': 0}
    remove_reasons = {}

    for filename in all_files:
        if not filename.endswith('.txt'):
            continue

        filepath = os.path.join(TEXTS_DIR, filename)
        filesize = os.path.getsize(filepath)

        # Very short files are fragments
        if filesize < 3000:
            remove.append((filename, 'too short (<3KB)'))
            stats['remove'] += 1
            stats['remove_bytes'] += filesize
            reason = 'too short (<3KB)'
            remove_reasons[reason] = remove_reasons.get(reason, 0) + 1
            continue

        # Check CSV bookshelf
        row = csv_data.get(filename, {})
        shelf = row.get('Bookshelf', '')
        shelf_verdict = classify_by_shelf(shelf)

        # Check filename patterns
        fn_verdict, fn_reason = classify_by_filename(filename)

        # Keep patterns override everything
        if fn_verdict == 'keep':
            keep.append(filename)
            stats['keep'] += 1
            stats['keep_bytes'] += filesize
            continue

        # Remove patterns
        if fn_verdict == 'remove':
            remove.append((filename, fn_reason))
            stats['remove'] += 1
            stats['remove_bytes'] += filesize
            remove_reasons[fn_reason] = remove_reasons.get(fn_reason, 0) + 1
            continue

        # Shelf-based removal
        if shelf_verdict == 'remove':
            reason = f'bookshelf: {shelf}'
            remove.append((filename, reason))
            stats['remove'] += 1
            stats['remove_bytes'] += filesize
            remove_reasons[reason] = remove_reasons.get(reason, 0) + 1
            continue

        # Default: keep
        keep.append(filename)
        stats['keep'] += 1
        stats['keep_bytes'] += filesize

    # Write output files
    with open('corpus_keep.txt', 'w') as f:
        for fn in keep:
            f.write(fn + '\n')

    with open('corpus_remove.txt', 'w') as f:
        for fn, reason in remove:
            f.write(f'{fn}\t{reason}\n')

    # Print summary
    print(f"Total files: {stats['keep'] + stats['remove']}")
    print(f"  Keep:   {stats['keep']:5d} files  ({stats['keep_bytes']/1e9:.2f} GB)")
    print(f"  Remove: {stats['remove']:5d} files  ({stats['remove_bytes']/1e9:.2f} GB)")
    print(f"  Removal rate: {stats['remove']*100/(stats['keep']+stats['remove']):.1f}%")
    print()
    print("Removal reasons:")
    for reason, count in sorted(remove_reasons.items(), key=lambda x: -x[1]):
        print(f"  {count:4d}  {reason}")


if __name__ == '__main__':
    main()
