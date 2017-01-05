import logging
import tables
from pandas.io.pytables import HDFStore

from selflearner.data_load.config_loader import Config
from selflearner.data_load.hdf5.pytables_descriptions import ConfigDescriptionOulad


class PytablesHdf5Manager:
    def __init__(self, file_path):
        logging.debug("Initialising PyTablesHdf5Manager")
        self.file_path = file_path
        self.logger = logging.getLogger(__name__)

    def check_exist_dataframes(self, key_array):
        """
        Checks whether all the provided keys exist in the store.

        :param key_array:
        :return: True if all of them exist otherwise False
        """
        with HDFStore(self.file_path) as store:
            for ds in key_array:
                if ds not in store:
                    logging.debug("%s not in hdf5 file", ds)
                    return False
                else:
                    logging.debug("%s checked", ds)
            return True

    def store_dataframe(self, key, df):
        try:
            key = self._check_and_get_alias(key)
        except OSError:
            pass
        except tables.exceptions.NoSuchNodeError:
            pass
        logging.debug("Storing dataframe: %s", key)

        with HDFStore(self.file_path) as store:
            store[key] = df
        logging.debug("DataFrame stored: %s", key)
        return

    # def load_dataframe(self, name):
    #     """
    #     Load specified table into pandas DataFrame
    #     :param name: table name
    #     :return: pandas.DataFrame with the table data
    #     """
    #     with tables.open_file(self.datastore_path, 'r') as ds:
    #         table = getattr(ds.root, name)
    #         return pandas.DataFrame.from_records(table.read())#

    def load_dataframe(self, key):
        key = self._check_and_get_alias(key)
        with HDFStore(self.file_path) as store:
            return store[key]

    def store_table_one_row(self, object, key, description):
        group_path, simple_key = self.create_groups(key)
        with tables.open_file(self.file_path, 'a') as ds:
            # setattr(ds.root,)
            self.logger.debug("Creating table with one row")
            self.logger.debug("GroupPath: %s, key: %s", group_path, simple_key)
            self._remove_node(ds, key)
            table = ds.create_table(group_path, simple_key,
                                    description=description,
                                    expectedrows=1)
            self.logger.debug("Table created %s", key)
            self.logger.debug("Storing table data")
            row = table.row
            row = self.append_object(row, object)
            self.logger.debug("Table stored")
            table.flush()

    def create_groups(self, key):
        group_arr = [g for g in key.split(sep='/') if g]
        logging.debug("Group arr: %s key:%s", group_arr, key)
        simple_key = group_arr[-1]
        with tables.open_file(self.file_path, 'a') as ds:
            group_node = ds.root
            group_path = '/'
            if len(group_arr) > 1:
                # return group_node, simple_key
                for g in group_arr[:-1]:
                    group_path = group_path + g + '/'
                    logging.debug("Creating new group node: %s in %s", g, group_node)
                    try:
                        group_node = ds.create_group(group_node, g)
                    except tables.exceptions.NodeError:
                        logging.debug("This node is already created: %s", g)
                        group_node = getattr(group_node, g)
            return group_path, simple_key

    def load_object_one_row(self, key):
        """
        Loads the object from the storage or returns None if it's not in present in the storage.
        :param key:
        :return:
        """
        self.logger.debug("Retrieving object by key: %s", key)
        try:
            with tables.open_file(self.file_path, 'r') as ds:
                table = getattr(ds.root, key)
                for row in table.iterrows(0,1):
                    self.logger.debug("Retrieved row: %s", row)
                    self.logger.debug("Retrieved row: %s", row['cutoff_date_train'])
                    return row
        except tables.exceptions.NoSuchNodeError:
            return None
        except IOError:
            logging.warning("File does not exist %s", self.file_path)
            return None

    def append_object(self, row, object):
        attr_names = [att for att in dir(object) if not att.startswith('__') and not callable(getattr(object, att))]
        self.logger.debug("Appending object to row")
        for att in attr_names:
            value = getattr(object, att)
            self.logger.debug("Inserting %s into row, value= %s ", att, value)
            row[att] = value
        self.logger.debug("Stored row: %s", row)
        self.logger.debug("Train:%s", row['id_assessment_train'])
        self.logger.debug("Test:%s", row['id_assessment_test'])
        row.append()
        return row

    def _check_and_get_alias(self, key):
        """
        Checks whether the name is alias and returns the target.
        :param key:
        """
        with tables.open_file(self.file_path, 'r') as ds:
            val = getattr(ds.root, key)
            val_type = type(val)
            if val_type is tables.link.SoftLink:
                key = val.target
                self.logger.debug("SoftLink: %s", key)
        return key

    def _get_group_simple_key(self, key):
        group_arr = [k for k in key.split(sep='/') if k]
        group_path = '/'
        if len(group_arr) > 1:
            group_path = group_path + '/'.join(group_arr[:-1])
        simple_key = group_arr[-1]
        return group_path, simple_key

    def create_alias(self, source, target):
        group_path, simple_key = self.create_groups(source)
        if not target.startswith('/'):
            target = '/' + target
        logging.debug("GroupPath: %s name: %s", group_path, simple_key)
        logging.debug("Creating link: %s -> %s", source, target)
        with tables.open_file(self.file_path, 'a') as ds:
            try:
                alias = ds.get_node(group_path, simple_key)
                logging.debug("Node soft_link already exists")
            except tables.exceptions.NoSuchNodeError:
                alias = ds.create_soft_link(group_path, simple_key, target=target)
                logging.debug("New node created")
            return alias

    def _remove_node(self, ds, name):
        """
        Remove node from datastore
        :param ds: pointer to datastore to remove node from
        :param name: node name
        :return: None
        """
        try:
            self.logger.debug('Trying to remove node %s', name)
            ds.remove_node(ds.root, name=name)
            self.logger.debug('Node %s was removed', name)
        except tables.NoSuchNodeError:
            self.logger.debug('Node %s not found', name)

    def print_tables(self):
        with tables.open_file(self.file_path, 'r') as ds:
            print(ds)

    def print_pandas(self):
        with HDFStore(self.file_path) as ds:
            print(ds)


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    file_path = 'test.h5'
    manager = PytablesHdf5Manager(file_path)

    logging.debug("Testing storing/loading config object")
    config = Config()
    config.assessment_name = 'TMA2'
    config.id_assessment = 4
    print(vars(config))
    manager.store_table_one_row(config, 'key', ConfigDescriptionOulad)
    with tables.open_file(file_path, 'r') as ds:
        print(ds)
        print(ds.root.key)

    row = manager.load_object_one_row('key')
    # logging.debug(row)
    config = Config.from_pytable_row(row)
    logging.debug("Config: %s", vars(config))

    # group_name = 'a/b/c'
    # logging.debug("Testing creating hiearchical group: %s", group_name)
    # with tables.open_file(file_path, 'a') as ds:
    #     group_arr = [g for g in group_name.split(sep='/') if g]
    #     group_node = ds.root
    #     try:
    #         ds.remove_node(group_node, group_arr[0], recursive=True)
    #     except tables.exceptions.NoSuchNodeError:
    #         logging.debug("Node does not exist:%s", group_arr[0])
    #     for g in group_arr:
    #         logging.debug("Creating new group node: %s in %s", g, group_node)
    #         try:
    #             group_node = ds.create_group(group_node, g)
    #         except tables.exceptions.NodeError:
    #             logging.debug("This node is already created: %s", g)
    #             group_node = getattr(group_node, g)
    logging.debug("New group created successfully")


if __name__ == "__main__":
    main()
