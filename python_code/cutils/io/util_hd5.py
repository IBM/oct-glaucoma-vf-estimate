import h5py


class HD5Util:
    def __init__(self, file_name, mode):
        """

        :param file_name:
        :param mode: 'r' or 'w'
        """
        self.mode = mode
        assert mode == 'r' or mode == 'w', 'mode should be one of r/w'
        self.f = h5py.File(file_name, mode)

    def add(self, key, data, more=False, gzip=False):

        """
        :param key:
        :param data:
        :param more: If False, closes the hd5 stream
        :param gzip if True, compresses the data using zip
        :return:
        """

        assert self.mode == 'w', 'File should be opened in write mode'
        if (gzip):
            self.f.create_dataset(key, data=data, compression="gzip", compression_opts=4)
        else:
            self.f.create_dataset(key, data=data)
        if (not more):
            self.f.close()

    def read(self, key):

        assert self.mode == 'r', 'File should be opened in read mode'

        data = self.f.get(key)

        return data

    def get_keys(self):

        return self.f.keys()

    def close(self):
        """
        Closes the  hd5 stream
        :return:
        """

        self.f.close
