"""This code split the labaled data into two sets, one for training,
    and one for the validation. In addition, the code returns the test_set
    """

from pathlib import Path
from sklearn.model_selection import train_test_split

full_data = ['ES2002', 'ES2005', 'ES2006', 'ES2007', 'ES2008', 'ES2009', 
                'ES2010', 'ES2012', 'ES2013', 'ES2015', 'ES2016', 'IS1000', 'IS1001',
                'IS1002', 'IS1003', 'IS1004', 'IS1005', 'IS1006', 'IS1007', 'TS3005',
                'TS3008', 'TS3009', 'TS3010', 'TS3011', 'TS3012']

def flatten(list_of_list):
    return [item for sublist in list_of_list for item in sublist]

full_data = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in full_data])
full_data.remove('IS1002a')
full_data.remove('IS1005d')
full_data.remove('TS3012c')

# Spliting to training and validation set
training_set, validation_set = train_test_split(full_data, test_size=0.3, random_state=1)

# Our spliting, that we have used for training
training_set = ['TS3005a', 'IS1003b', 'TS3012d', 'IS1000a', 'IS1002d', 'ES2009d', 'ES2013c', 'TS3009b', 'IS1001a', 'IS1005c', 'ES2007d', 'TS3008c', 'ES2016b', 'TS3010c', 'ES2010c', 'TS3012b', 'ES2016d', 'IS1006d', 'ES2005a', 'IS1002c', 'IS1001b', 'ES2009b', 'TS3009c', 'ES2002d', 'TS3009d', 'ES2012c', 'IS1004d', 'ES2016c', 'IS1000d', 'IS1001d', 'IS1007d', 'ES2010a', 'ES2006a', 'ES2008b', 'IS1004b', 'ES2002a', 'TS3010b', 'IS1003c', 'ES2009c', 'IS1004c', 'IS1005a', 'ES2005d', 'TS3011d', 'ES2007b', 'IS1006c', 'TS3009a', 'ES2007c', 'ES2012b', 'ES2012a', 'ES2006d', 'ES2008c', 'ES2009a', 'IS1001c', 'ES2010b', 'ES2005c', 'IS1007b', 'TS3005c', 'ES2002b', 'ES2008a', 'IS1005b', 'TS3008b', 'ES2005b', 'TS3005b', 'ES2006b', 'IS1007c', 'ES2007a', 'ES2015b']
validation_set = ['ES2013b', 'IS1003d', 'ES2013a', 'TS3011a', 'IS1000b', 'IS1003a', 'IS1006a', 'TS3010a', 'TS3010d', 'IS1004a', 'TS3008d', 'IS1006b', 'TS3008a', 'IS1000c', 'TS3005d', 'ES2010d', 'IS1007a', 'ES2016a', 'ES2006c', 'ES2002c', 'ES2015d', 'ES2012d', 'IS1002b', 'ES2008d', 'TS3011b', 'ES2015c', 'ES2013d', 'ES2015a', 'TS3011c', 'TS3012a']


# Test data 
test_set = ['ES2003', 'ES2004', 'ES2011', 'ES2014', 'IS1008', 'IS1009', 'TS3003', 'TS3004', 'TS3006', 'TS3007']
test_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in test_set])



def training_validation_test_sets() :
    return training_set, validation_set, test_set

