import numpy
import os


for root, direc, filez in os.walk(os.path.join('..', '..', 'unaware_semantics_bids', 'events_other_formats')):
    for f in filez:
        if 'events' in f:
            out_folder = root.replace('events_other_formats', 'sourcedata')
            assert os.path.exists(out_folder)
            if 'uncorrected' in f:
                ### relevant file
                output_name = f.replace('_uncorrected_RTs', '')
                file_dict = dict()
                with open(os.path.join(root, f)) as i:
                    counter = 0
                    for l in i:
                        line = l.strip().split('\t')
                        if counter == 0:
                            header = line.copy()
                            file_dict = {h : list() for h in header}
                            counter += 1
                            continue
                        for h_i, h in enumerate(header):
                            file_dict[h].append(line[h_i])
                ### correcting accuracy
                file_dict['response_time_corrected'] = [float(v)-float(file_dict['PAS_RT'][v_i]) for v_i, v in enumerate(file_dict['response_time'])]
                ### correcting PAS RT
                ### 10 frames, 0.16*frame
                fixed_correction = 1. + 1. + 0.166
                #(numpy.nanmean([float(v) for v in file_dict['duration']])*5)
                #print(fixed_correction)
                file_dict['PAS_RT_corrected'] = [float(v)-float(file_dict['response_time'][v_i-1])-fixed_correction if v_i!=0 else 'na' for v_i, v in enumerate(file_dict['PAS_RT'])]
                del file_dict['PAS_RT']
                file_dict['PAS_RT'] = file_dict['PAS_RT_corrected'].copy()
                del file_dict['PAS_RT_corrected']
                del file_dict['response_time']
                file_dict['response_time'] = file_dict['response_time_corrected'].copy()
                del file_dict['response_time_corrected']
                total_lengths = [len(v) for v in file_dict.values()]
                assert len(set(total_lengths)) == 1
                total_length = list(set(total_lengths))[0]
                print(os.path.join(out_folder, output_name))
                with open(os.path.join(output_folder, output_name), 'w') as o:
                    for h in header:
                        o.write('{}\t'.format(h))
                    o.write('\n')
                    for l_i in range(total_length):
                        for h in header:
                            o.write('{}\t'.format(file_dict[h][l_i]))
                        o.write('\n')


