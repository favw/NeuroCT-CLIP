# -*- coding: utf-8 -*-

# File has to be called with INPUT_PATH and OUTPUT_PATH. INPUT_PATH leads to a
# directory containing subdirectories {PATIENT_NAME} with radiology reports in PDF format.
import os
import argparse
import csv 
from pathlib import Path
from text_filter import parse_text

C_FLAG = False 
MAP_FLAG = False
REPORT_CSV_DATA_FIELDS = ['Name', 'Clinical_Information', 'Technique', 'Findings', 'Impressions']
LABELS_CSV_DATA_FIELDS = ['Acute subdural hematoma', 'Chronic subdural hematoma',
                          'Epidural hematoma', 'Subarachnoid hemorrhage', 'Intracerebral Bleeding',
                          'Calvarial fracture', 'Basilar skull fracture', 'Fracture/other', 'Vascular sclerosis', 'Reduced Brain Volume', 
                          'Normal Pressure Hydrocephalus', 'Hydrocephalus', 'Cerebral Edema','Shunt leftovers', 'Shunt', 
                          'Chronic Infarction', 'Subacute Infarction', 'Acute Infarction', 'Aneurysma', 'Cavernoma', 
                          'Meningeoma', 'Glioblastoma', 'Pituitary Adenoma', 'Tumor/other', 'Mastoiditis', 'Mastoidectomy', 'EVD', 'Trepanation',
                          'Healthy', 'Midline Shift', 'Ventricle Compression', 'Increased Ventricle Width', 'Enlargement of outer liquor space',
                          'Tissue Defect', 'Cavum vergae', 'Superficial Haematoma', 'Sinus Blockage', 'Microangiopathy',
                          'Macroangiopathy', 'Brain herniation', 'Pseudophakia', 'Middle ear effusion', 'Lacunary Infarction',
                          'Abscess', 'Abnormal Liquor', 'Meningitis', 'Increased Intracranial Pressure', 'Empyema', 
                          'Contrast medium', 'Enlarged perivascular space', 'Jugular bulb diverticulum', 'Trapped air',
                          'Hypodense area', 'Cerumen', 'Medical Material / other', 'Vascular Clip', 'vertebrobasilar dolichoectasia',
                          'Agenesis of the corpus callosum', 'Cyst/Hygroma', 'spinal decompression', 'onodi cell',
                          'Movement artifacts', 'Megacisterna magna', 'Chiari Malformation', 'Hemicraniotomy'
] 
METADATA_CSV_DATA_FIELDS = [] 

def main():
    global C_FLAG
    global MAP_FLAG
    args = getArgs()
    C_FLAG = args.c
    MAP_FLAG = args.map

    print("parsed Arguments...")
    input_path = Path(args.INPUT_PATH).resolve()
    output_path = Path(args.OUTPUT_PATH).resolve()
    if input_path == output_path:
        raise ValueError("INPUT_PATH and OUTPUT_PATH must differ.")

    output_path.mkdir(parents=True, exist_ok=True)
    reports = getReports(input_path)
    
    print("Found " + len(reports).__str__() + " reports.")
    """
    print("They are named:")
    for file in reports:
        print(file.__str__())
    """
    #reports_txt contains Files directing to .txt data
    reports_txt = convertReports(reports, input_path, output_path)
    print("Found " + len(reports).__str__() + " converted reports.") 
    """
    print("They are named:")
    for file in reports_txt:
        print(file.__str__())
    """
    print("Filtering " + len(reports).__str__() + " files...")
    filtered_files = [] 
    for file in reports_txt:
        parsed_text = parse_text(file)
        file.with_name(file.stem + "_FILTERED.txt").write_bytes(("".join(parsed_text)).encode())
        filtered_files.append(parsed_text)
        
    print("Removed Hospital and Patient Data")

    print("Writing reports and labels .csv...")
    impressions = writeReportCSV(output_path, filtered_files, reports_txt)
   #print("in file: " + os.path.join(path, "UNANOMYNIZED_REPORTS.csv").__str__())
    if MAP_FLAG:
        createIdentifier(output_path)
    writeImpressionsCSV(output_path, impressions)
    

def createIdentifier(path):
    if path.is_dir():
        if C_FLAG:
            exist_flag = False 
            anonymized_map = []
            if os.path.isfile(os.path.join(path, "ANONYMIZED_MAP.csv")): exist_flag = True
            existing_content = ""
            if exist_flag:
                with open(os.path.join(path, "ANONYMIZED_MAP.csv"), encoding='utf-8') as f:
                    existing_content = f.read()
            mode = 'a' if exist_flag else 'w'
            with open(os.path.join(path, "ANONYMIZED_MAP.csv"), mode, newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                i = 0
                if not exist_flag: writer.writerow(['anonymized_id', 'name_id'])
                else: 
                    i = max(len(existing_content.splitlines()) - 1, 0)
                for patient in path.iterdir():
                    patstr = os.path.basename(patient).__str__()
                    if patstr.endswith('.txt'):
                        if "FILTERED" in patstr:
                            name_id_res = patstr.split("_")
                            if name_id_res.__len__() > 0:
                                name_id = "" 
                                for cur in name_id_res:
                                    if not cur == "": name_id += cur + "_"
                                    else: break
                                name_id += name_id_res[-2]
                                if exist_flag: 
                                    if name_id not in existing_content:
                                        writer.writerow(["patient_" + str(i), name_id])
                                        anonymized_map.append("patient_" + str(i))
                                        i += 1  
                                else: 
                                    writer.writerow(["patient_" + str(i), name_id])
                                    anonymized_map.append("patient_" + str(i))
                                    i += 1
                            else: return
                return anonymized_map


# get list of radiology reports (.pdf!) in PATH 
def getReports(path):
    print("finding reports in " + path.__str__())
    reports = []
    if path.is_dir():
        if(C_FLAG):
            for patient in path.iterdir():
                    if(patient.__str__().endswith('.pdf')):
                        reports.append(os.path.realpath(Path(patient)))
        else: 
            for patient in path.iterdir():
                 if patient.is_dir():
                    for result in Path(patient).glob('*.pdf'):
                        reports.append(os.path.realpath(result))
    else:
        print("The given path is not a directory. Please open using -h for further instructions.")
    return reports


"""
Parse arguments and flags:
    PATH: given Path to directory containing folders with Patient data as folders DEFAULT: none
    -c: sets optional flag to make PATH link to a folder containing .pdf data DEFAULT: False
"""
def getArgs():
    parser = argparse.ArgumentParser(description="Python Script for Conversion of PDF-Files to .csv")

    parser.add_argument(
        "INPUT_PATH",
        type=str, 
        help="MANDATORY: Set path to directory containing subdirectories (Patient) which contain .pdf radiology-reports")
    parser.add_argument(
        "OUTPUT_PATH",
        type=str,
        help="MANDATORY: Set path to a different directory where converted text and CSV files will be written")
    parser.add_argument(
        "-c", 
        action='store_true', 
        help = "optional flag to make {INPUT_PATH} link to a folder containing .pdf data")
    parser.add_argument(
        "-map",
        action='store_true',
        help = "set this flag to create a .csv containing patient names mapped to a unique identifier. The unique identifier will be used in the .csv containing labels"
    )

    return parser.parse_args() 

# input: List of links to radiology report files as .pdf
# output: List of links to radiology report files as .txt
# creates .txt files in the output folder, mirroring input subfolders when present.
def convertReports(reports, input_path, output_path):
    import pymupdf

    convertedReports = []
    for report in reports:        
         report_path = Path(report)
         with (pymupdf.open(report_path) as file):
            text = chr(12).join([page.get_text() for page in file]) 
         output_report = output_path / report_path.relative_to(input_path).with_suffix(".txt")
         output_report.parent.mkdir(parents=True, exist_ok=True)
         output_report.write_bytes(text.encode())
         convertedReports.append(output_report)
    return convertedReports




def writeReportCSV(path, filtered_text, path_lists):
    """
    Parameters
    path: Path to target file
    filtered_text: List containing filtered text-string in path_lists
    path_lists: List of Paths to files containing the orginal text-string
    """
    impressions_list = []
    with open(os.path.join(path, "UNANOMYNIZED_REPORTS.csv"), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(REPORT_CSV_DATA_FIELDS)
        # list containing (name, impression) tuples 
        for text, path in zip(filtered_text,path_lists):
            categorized_text = getDataFromText(text, path)
            impressions_list.append((categorized_text[0], categorized_text[4]))
            writer.writerow(categorized_text)

    return impressions_list

# input: findings list containing (name, findings) tuples 
def writeImpressionsCSV(path, findings):
    with open(os.path.join(path, "UNANOMYNIZED_LABELS.csv"), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(LABELS_CSV_DATA_FIELDS)
        for name, findings in findings:
            result = [0] * LABELS_CSV_DATA_FIELDS.__len__()
            result.insert(0, name)
            writer.writerow(result)

# filter textual data using given delimiters 
# returns [name, clinical info, technique, findings, impressions]
# TODO: make delimiters variable 
def getDataFromText(text, path):
    CLINICAL_INFORMATION = []
    TECHNIQUE = []
    FINDINGS = []
    IMPRESSIONS = []
    findings_flag = False 
    impressions_flag = False
    for line in text:
        if findings_flag:
            if not ("Beurteilung" in line):
                FINDINGS.append(line)
            else: 
                findings_flag = False
                impressions_flag = True
                continue
        if impressions_flag:
            IMPRESSIONS.append(line)
        if "Klinische Angaben:" in line: 
            CLINICAL_INFORMATION.append(line.split("Klinische Angaben:")[1])
        if "Native" in line:
            TECHNIQUE.append(line)
            findings_flag = True
    return [os.path.basename(path).__str__().split(".txt")[0], "".join(CLINICAL_INFORMATION), "".join(TECHNIQUE), "".join(FINDINGS), "".join(IMPRESSIONS)]


if __name__ == '__main__':
    main()








