import tables
from tables import IsDescription


class ConfigDescriptionOulad(IsDescription):
    cutoff_date_train = tables.Int32Col()
    cutoff_date_test = tables.Int32Col()
    pres_start = tables.Int32Col()
    id_assessment_train = tables.Int32Col()
    id_assessment_test = tables.Int32Col()
    assessment_name = tables.StringCol(16)
    days_to_cutoff = tables.Int32Col()
    current_date_train = tables.Int32Col()
    current_date_test = tables.Int32Col()
    train_labels_from = tables.Int32Col()
    train_labels_to = tables.Int32Col()
    test_labels_from = tables.Int32Col()
    test_labels_to = tables.Int32Col()


class ConfigDescriptionLive(IsDescription):
    cutoff_date = tables.Time32Col()
    pres_start = tables.Time32Col()
    id_assessment = tables.Int32Col()
    assessment_name = tables.StringCol(16)
    days_to_cutoff = tables.Int32Col()
    current_date = tables.Time32Col()
    train_labels_from = tables.Time32Col()
    train_labels_to = tables.Time32Col()
    test_labels_from = tables.Time32Col()
    test_labels_to = tables.Time32Col()
