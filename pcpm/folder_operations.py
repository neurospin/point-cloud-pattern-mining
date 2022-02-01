from os import path, getcwd, mkdir, sep
import shutil
from collections import namedtuple
import anytree
from glob import glob

import logging
log = logging.getLogger(__name__)

class _ExperimentFolderStructure:
    def __init__(self):
        """
        The abstract folder structure
        """
        self.experiment = anytree.Node('experiment', id='experiment')
        self.data_processing = anytree.Node("data_processing", parent=self.experiment, id='data_processing')
        self.isomap = anytree.Node('isomap', parent=self.data_processing, id='isomap')
        self.region = anytree.Node('region', parent=self.isomap, id='region')
        self.spam = anytree.Node("SPAM", parent=self.data_processing, id='spam')
        self.region_spam = anytree.Node('region', parent=self.spam, id='region_spam')
        self.region_spam_ICP = anytree.Node('ICP', self.region_spam, id='region_spam_ICP')
        self.ICP = anytree.Node('tal', parent=self.spam, id='ICP')
        # self.transform = anytree.Node('transform', parent=self.experiment, id='transform')
        # self.subjectNames = anytree.Node('subject_names', parent=self.experiment, id='subject_names')

    def __repr__(self):
        return anytree.RenderTree(self.data_processing).__str__()

class _ExperimentFolderPath(_ExperimentFolderStructure):
    def __init__(self, experiment_path, region):
        """
        The folder structure of this experiment
        """
        _ExperimentFolderStructure.__init__(self)
        if path.isabs(experiment_path):
            root, experiment_path = path.split(experiment_path)
        else:
            root = getcwd()[1:] # getcwd()[1:] is the cwd without the first slash
        # create a root node representing the CWD and
        # attach the folder structure to this node
        self.root = anytree.Node(root, id='root')

        self.experiment.parent = self.root
        self.region.name = region
        self.region_spam.name = region
        self.experiment.name = experiment_path
    def __repr__(self):
        s = ''
        for pre, _, node in anytree.RenderTree(self.root):
             s += ("%s%s\n" % (pre, node.name))
        return s

    def as_dictionary(self):
        """
        Get the absolute paths as dictionary
        """
        path_dict = dict()
        for node in anytree.iterators.preorderiter.PreOrderIter(self.root):
            path_dict[node.id] = node.separator.join([""] + [str(cur_node.name) for cur_node in node.path])
        return path_dict

    def as_namedtuple(self):
        """
        Get the absolute paths as namedtuple
        """
        path_dict = self.as_dictionary()
        return namedtuple("Paths", path_dict)(**path_dict)


def create_folders(region, experiment_folder):
    """
    Create an empty folder structure for the project.
    """
    log.info("Create processing folder structure")
    dp = _ExperimentFolderPath(region, experiment_folder)
    log.info("n"+str(dp))
    for folder in dp.as_dictionary().values():
        if not path.exists(folder):
            mkdir(folder)
    return dp.as_namedtuple()


def copy_trm_files(source_path, dest_path):
    """
    Copy the .trm files from source_path to the
    """
    log.info("Copy .trm files into the processing folder")
    file_paths = glob(path.join(source_path,'*.trm'))
    for file_path in file_paths:
        basename = path.basename(file_path)
        # copy file and add L prefix
        shutil.copy(
            file_path,
            path.join(dest_path, "L%s" % basename),
            follow_symlinks=False)
        # copy file and add R prefix
        shutil.copy(
            file_path,
            path.join(dest_path, "R%s" % basename),
            follow_symlinks=False)

if __name__ == "__main__":

    dp = _ExperimentFolderPath(
        region='region',
        experiment_path='my_experiment'
    )
    
    dp = create_folders('my_experiment', 'region')
    print(dp)
    # copy_trm_files("./example_ophelie/exp_CSSyl_Chimps_30/tal", dp.ICP)

