osr_splits = {
    # ['01_TUMOR', '02_STROMA',  '03_COMPLEX', '04_LYMPHO', '05_DEBRIS',  '06_MUCOSA', '07_ADIPOSE', '08_EMPTY']
    'kather2016': [
        [0, 1, 2, 3,    5    ],    # S0: unknown: '05_DEBRIS',              '07_ADIPOSE', '08_EMPTY' uninformative according to Jakob
        [0, 1, 2,            ],    # S2: known: tumor, stroma (no immune/tumoral cells), complex stroma (immune/tumoral cells + stroma) (interesting because complex can contain immune)
        [0, 1, 2, 3          ],    # S4: known: tumor, stroma (no immune/tumoral cells), complex stroma (immune/tumoral cells + stroma), lympho (immune cells, no stroma)
        ],
    ####################################################################################################################
    # KATHER_100K
    # 01_ADIPOSE  02_BACKGROUND  03_DEBRIS  04_LYMPHOCITES  05_MUCUS  06_MUSCLE  07_NORMAL  08_STROMA  09_TUMOR
    # Adipose (ADI), background (BACK), debris (DEB), lymphocytes (LYM), mucus (MUC), smooth muscle (MUS),
    # normal colon mucosa (NORM), cancer-associated stroma (STR), colorectal adenocarcinoma epithelium (TUM)
    'kather100k': [
        [         3,    5, 6, 7, 8], # S0 unknown: 01_ADIPOSE,02_BACKGROUND,03_DEBRIS,05_MUCUS, uninformative according to Jakob
        [                     7, 8], # S2 known: stroma and tumor
        [               5 ,   7, 8], # S3 known: muscle, stroma and tumor
     ],
}

