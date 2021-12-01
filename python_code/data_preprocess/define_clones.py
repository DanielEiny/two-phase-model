import os
import time
import subprocess
import pandas as pd


creat_germline_fields = ["germline_v_call", 
                         "germline_d_call", 
                         "germline_j_call", 
                         "germline_alignment", 
                         "germline_alignment_d_mask", 
                         "germline_alignment_v_region", 
                         "germline_regions"]

irrelevant_fields = ["v_call_original",
                     "v_call_genotypedת",
                     "v_cigar",
                     "d_cigar",
                     "j_cigar",
                     "v_score",
                     "v_identity",
                     "v_support",
                     "d_score",
                     "d_identity",
                     "d_support",
                     "j_score",
                     "j_identity",
                     "j_support",
                     "fwr1",
                     "fwr2",
                     "fwr3",
                     "fwr4",
                     "cdr1",
                     "cdr2",
                     "cdr3",
                     "sequence_alignment_aa",
                     "germline_alignment_aa",
                     "v_sequence_alignment_aa",
                     "v_germline_alignment_aa",
                     "d_sequence_alignment_aa",
                     "d_germline_alignment_aa",
                     "j_sequence_alignment_aa",
                     "j_germline_alignment_aa",
                     "fwr1_aa",
                     "cdr1_aa",
                     "fwr2_aa",
                     "cdr2_aa",
                     "fwr3_aa",
                     "fwr4_aa",
                     "cdr3_aa",
                     "junction_aa_length",
                     "fwr1_start",
                     "fwr1_end",
                     "cdr1_start",
                     "cdr1_end",
                     "fwr2_start",
                     "fwr2_end",
                     "cdr2_start",
                     "cdr2_end",
                     "fwr3_start",
                     "fwr3_end",
                     "fwr4_start",
                     "fwr4_end",
                     "cdr3_start",
                     "cdr3_end",
                     "cregion",
                     "sort",
                     "isotypecounter...101",
                     "vgenecounter",
                     "jgenecounter",
                     "isotypecounter...104"]

required_fields = ["sequence_id",
                   "sequence_alignment", 
                   "v_call",
                   "d_call",
                   "j_call",
                   "v_sequence_start",
                   "v_sequence_end",
                   "v_germline_start",
                   "v_germline_end",
                   "d_sequence_start",
                   "d_sequence_end",
                   "d_germline_start",
                   "d_germline_end",
                   "j_sequence_start",
                   "j_sequence_end",
                   "j_germline_start",
                   "j_germline_end",
                   "np1_length",
                   "np2_length"]

int_fields = required_fields[5:]


def define_clones(sample_path,
                  output_dir,
                  consensus_count_above=2,
                  v_sequence_end_above=250):

    file_name = os.path.basename(sample_path)
    out_file_path = os.path.join(output_dir, file_name)
    os.makedirs(output_dir, exist_ok=True)
    sample_name = file_name.replace('_cloned_w_filtered_seqs.tsv', '')

    repertoire = pd.read_csv(sample_path, sep='\t')

    # --- filter data, drop irrelevant columns, and save to temporary file --- #
    print(f'{time.ctime()} | {sample_name}: Read & Filter ')
    repertoire = repertoire.drop(columns=creat_germline_fields + irrelevant_fields,
                                 errors='ignore')

    consensus_count_filter = repertoire.consensus_count > consensus_count_above
    v_length_filter = repertoire.v_sequence_end > v_sequence_end_above
    no_umi_filter = ~repertoire.sequence_id.str.contains('FAKE')
    required_fields_filter = ~repertoire[required_fields].isna().any(axis=1)
    nan_filter = ~repertoire.complete_vdj.isna()  # Occures in cancer set, 
                                                  # indicates copy sequences.
    filter_union = consensus_count_filter & \
                   v_length_filter & \
                   no_umi_filter & \
                   required_fields_filter & \
                   nan_filter

    repertoire = repertoire[filter_union]
    repertoire[int_fields] = repertoire[int_fields].astype(int)

    repertoire.to_csv(out_file_path, sep='\t', index=False)

    # --- Find optimal clone seperation distance using "shazam" --- #
    logfile = os.path.join(output_dir, 'log.tmp')
    f = open(logfile, 'w')
    print(f'{time.ctime()} | {sample_name}: Find clone split threshold ')
    try:
        result = subprocess.run(['nice',
                                 '-19',
                                 'Rscript',
                                 'r_code/find_clone_split_threshold.R', 
                                 out_file_path],
                                 timeout=3600,
                                 stderr=f,
                                 stdout=subprocess.PIPE)
        threshold = float(result.stdout.decode().split(' ')[-1])
    except:
        print(f'{time.ctime()} | {sample_name}: Faild! ')
        os.remove(out_file_path)
        return

    # --- run DefineClones.py --- #
    print(f'{time.ctime()} | {sample_name}: DefineClones ')
    try:
        subprocess.run(['nice', '-19',
                        'DefineClones.py',
                        '-d', out_file_path,
                        '--act', 'set',
                        '--model', 'ham',
                        '--norm', 'len',
                        '--dist', str(threshold),
                        '--nproc', '60'],
                        stderr=f,
                        stdout=f)
    except:
        print(f'{time.ctime()} | {sample_name}: Faild! ')
        os.remove(out_file_path)
        return

    # --- run CreateGermlines.py --- #
    print(f'{time.ctime()} | {sample_name}: CreateGermlines ')

    personal_genotype_dir = sample_path.replace('_cloned_w_filtered_seqs.tsv', 
                                                '_interm_files')
    genotyped_v_ref = os.path.join(personal_genotype_dir, 
                                   sample_name + '_V_personal_ref_gapped.fasta')
    genotyped_d_ref = genotyped_v_ref.replace('_V_personal_ref_gapped.fasta',
                                              '_D_personal_ref.fasta')
    genotyped_j_ref = genotyped_d_ref.replace('D', 'J')
    
    # > Patch for covid and cancer files < #
    #personal_genotype_arcive_name = sample_name + '_interm_files'
    #sample_dir = os.path.dirname(sample_path)
    #personal_genotype_arcive = os.path.join(sample_dir,
    #                                        sample_name, # For covid only
    #                                        personal_genotype_arcive_name)

    #genotyped_v_ref = os.path.join(personal_genotype_arcive_name, sample_name + '_V_personal_ref_gapped.fasta')
    #genotyped_d_ref = os.path.join(personal_genotype_arcive_name, sample_name + '_D_personal_ref.fasta')
    #genotyped_j_ref = os.path.join(personal_genotype_arcive_name, sample_name + '_J_personal_ref.fasta')

    #subprocess.run(['nice', '-19',
    #                'tar', '-xzvf',
    #                personal_genotype_arcive + '.tgz',
    #                '-C',
    #                output_dir,
    #                genotyped_v_ref,
    #                genotyped_d_ref ,
    #                genotyped_j_ref],
    #                stderr=f,
    #                stdout=f)
    #genotyped_v_ref = os.path.join(output_dir, genotyped_v_ref)
    #genotyped_d_ref = os.path.join(output_dir, genotyped_d_ref)
    #genotyped_j_ref = os.path.join(output_dir, genotyped_j_ref)
    # > end of patch < #


    try:
        subprocess.run(['nice', '-19',
                        'CreateGermlines.py',
                        '-d', out_file_path.replace(".tsv", "_clone-pass.tsv"),
                        '-g', 'dmask',
                        '-r', genotyped_v_ref, genotyped_d_ref, genotyped_j_ref,
                        '--cloned'],
                        stderr=f,
                        stdout=f)
    except:
        print(f'{time.ctime()} | {sample_name}: Faild! ')
        os.remove(out_file_path)
        return

    # --- Clean-up --- #
    print(f'{time.ctime()} | {sample_name}: Clean-up ')
    f.close()
    subprocess.run(['rm',
                    out_file_path.replace(".tsv", "_clone-pass.tsv")])

    subprocess.run(['mv',
                    out_file_path.replace(".tsv", "_clone-pass_germ-pass.tsv"),
                    out_file_path])
  

if __name__ == "__main__":
    define_clones("/work/peresay/vdjbase/V9/P4/P4_I10_S1/P4_I10_S1_cloned_w_filtered_seqs.tsv",
                  "/home/bcrlab/daniel/two-phase-model/data/P4")
