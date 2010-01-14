#!/usr/bin/python
#           
# Usage:
#       Run gcov on the results of "make alltests" and create tar ball containing coverage results for one machine
#           ./gcov.py -run_gcov
#       Generate html pages showing coverage by merging tar balls from multiple machines (index_gcov1.html and index_gcov2.html)
#           ./gcov.py -merge_gcov [LOC] tarballs
#


def run_gcov(petsc_dir,user,gcov_dir):

    # 1. Runs gcov 
    # 2. Saves the untested source code line numbers in 
    #    xxx.c.lines files in gcov_dir

    import os
    import string
    import shutil

    print "Creating directory to save .lines files\n"
    if os.path.isdir(gcov_dir):
        shutil.rmtree(gcov_dir)
    os.mkdir(gcov_dir)
    print "Running gcov\n"
    for root,dirs,files in os.walk(os.path.join(PETSC_DIR,"src")):
        # Exclude tests and tutorial directories
        if root.endswith('tests') | root.endswith('tutorials'):
            continue
        os.chdir(root)
        for file_name in files:
            csrc_file = file_name.endswith('.c')
            if csrc_file:
                c_file = file_name.split('.c')[0]
                gcov_graph_file = c_file+".gcno"
                gcov_data_file  = c_file+".gcda" 
                if os.path.isfile(os.path.join(gcov_graph_file)) & os.path.isfile(os.path.join(gcov_data_file)):
                    # gcov created .gcno and .gcda files => create .gcov file,parse it and save the untested code line
                    # numbers in .lines file
                    os.system("gcov "+file_name)
                    gcov_file = file_name+".gcov"
                    try:
                        gcov_fid = open(gcov_file,'r')
                        root_tmp1 = 'src'+root.split("src")[1].replace(os.sep,'_')
                        lines_fid = open(os.path.join(gcov_dir,root_tmp1+'_'+file_name+'.lines'),'w')
                        for line in gcov_fid:
                            if line.find("#####") > 0:
                                line_num = line.split(":")[1].strip()
                                print >>lines_fid,"""%s"""%(line_num)
                        gcov_fid.close()
                        lines_fid.close()
                    except IOError:
                        continue
                    else:
                        # gcov did not create .gcno or .gcda file,save the source code line numbers to .lines file
                        file_id = open(file_name,'r')
                        root_tmp1 = 'src'+root.split("src")[1].replace(os.sep,'_')
                        lines_fid = open(os.path.join(gcov_dir,root_tmp1+'_'+file_name+'.lines'),'w')
                        nlines = 0
                        line_num = 1
                        for line in file_id:
                            if line.strip() == '':
                                line_num += 1
                            else:
                                print >>lines_fid,"""%s"""%(line_num)
                                line_num += 1
                        file_id.close()
                        lines_fid.close()
    print """End of script"""                                
    return

def make_tarball(gcov_dir):

    # Create tarball of .lines files stored in gcov_dir
    import os
    import shutil
    print """Creating tarball in %s\n""" %(gcov_dir)
    os.chdir(gcov_dir)
    os.system("tar -czf "+PETSC_DIR+"/gcov.tar.gz *.lines")
    shutil.rmtree(gcov_dir)
    print """Tarball created\n"""
    return

def make_htmlpage(loc,tarballs):

    # Create index_gcov webpages using information processed from
    # running gcov
    # This is done in four stages
    # Stage 1: Extract tar balls,merge files and dump .lines files in /tmp/gcov
    # Stage 2: Process .lines files
    # Stage 3: Create marked HTML source code files
    # Stage 4: Create HTML pages having statistics and hyperlinks to HTML source code           files (files are sorted by filename and percentage code tested) 
    #  Stores the main HTML pages in LOC if LOC is defined via command line argument o-wise it uses the default PETSC_DIR
    
    import os
    import string
    import operator
    import sys
    import shutil
    import glob
    from time import gmtime,strftime

    PETSC_DIR = os.environ['PETSC_DIR']
    gcov_dir = os.path.join('/tmp','gcov')
    if os.path.isdir(gcov_dir):
        shutil.rmtree(gcov_dir)
    os.makedirs(gcov_dir)
    cwd = os.getcwd()
    # -------------------------- Stage 1 -------------------------------
    len_tarballs = len(tarballs)
    if len_tarballs == 0:
        print "No gcov tar balls found in directory %s" %(cwd)
        sys.exit()

    print "%s tarballs found\n%s" %(len_tarballs,tarballs)
    print "Extracting gcov directories from tar balls"
    #  Each tar file consists of a bunch of *.line files NOT inside a directory
    tmp_dirs = []
    for i in range(0,len_tarballs):
        tmp = []
        dir = os.path.join(gcov_dir,str(i))
        tmp.append(dir)
        os.mkdir(dir)
        os.system("cd "+dir+";gunzip -c "+cwd+"/"+tarballs[i] + "|tar -xof -")
        tmp.append(len(os.listdir(dir)))
        tmp_dirs.append(tmp)

    # each list in tmp_dirs contains the directory name and number of files in it           
    # Cases to consider for gcov
    # 1) Gcov runs fine on all machines = Equal number of files in all the tarballs.         
    # 2) Gcov runs fine on atleast one machine = Unequal number of files in the tarballs.The smaller tarballs are subset of the largest tarball(s)   
    # 3) Gcov doesn't run correctly on any of the machines...possibly different files in tarballs  

    # Case 2 is implemented for now...sort the tmp_dirs list in reverse order according to the number of files in each directory
    tmp_dirs.sort(key=operator.itemgetter(1),reverse=True)

    # Create temporary gcov directory to store .lines files
    print "Merging files"
    nfiles = tmp_dirs[0][1]
    files_dir1 = os.listdir(tmp_dirs[0][0])
    for i in range(0,nfiles):
        out_file = os.path.join(gcov_dir,files_dir1[i])
        out_fid  = open(out_file,'w')

        in_file = os.path.join(tmp_dirs[0][0],files_dir1[i])
        in_fid = open(in_file,'r')
        lines = in_fid.readlines()
        in_fid.close()
        for j in range(1,len(tmp_dirs)):
            in_file = os.path.join(tmp_dirs[j][0],files_dir1[i])
            try:
                in_fid = open(in_file,'r')
            except IOError:
                continue
            new_lines = in_fid.readlines()
            lines = list(set(lines)&set(new_lines)) # Find intersection             
            in_fid.close()

        if(len(lines) != 0):
            lines.sort()
            out_fid.writelines(lines)
            out_fid.flush()

        out_fid.close()

    # Remove directories created by extracting tar files                                                                                 
    print "Removing temporary directories"
    for j in range(0,len(tmp_dirs)):
        shutil.rmtree(tmp_dirs[j][0])

    # ------------------------- End of Stage 1 ---------------------------------

    # ------------------------ Stage 2 -------------------------------------
    print "Processing .lines files in %s" %(gcov_dir)
    gcov_filenames = os.listdir(gcov_dir)
    nsrc_files = 0; 
    nsrc_files_not_tested = 0;
    src_not_tested_path = [];
    src_not_tested_filename = [];
    src_not_tested_lines = [];
    src_not_tested_nlines = [];
    ctr = 0;
    print "Processing gcov files"
    for file in gcov_filenames:
        tmp_filename = string.replace(file,'_',os.sep)
        src_file = string.split(tmp_filename,'.lines')[0]
        gcov_file = gcov_dir+os.sep+file
        gcov_fid = open(gcov_file,'r')
        nlines_not_tested = 0
        lines_not_tested = []
        for line in gcov_fid:
            nlines_not_tested += 1
            temp_line1 = line.lstrip()
            temp_line2 = temp_line1.strip('\n')
            lines_not_tested.append(temp_line2)
        if nlines_not_tested :   
            nsrc_files_not_tested += 1
            k = string.rfind(src_file,os.sep)
            src_not_tested_filename.append(src_file[k+1:])
            src_not_tested_path.append(src_file[:k])
            src_not_tested_lines.append(lines_not_tested)
            src_not_tested_nlines.append(nlines_not_tested)
        nsrc_files += 1
        gcov_fid.close()

    # ------------------------- End of Stage 2 --------------------------

    # ---------------------- Stage 3 -----------------------------------
    print "Creating marked HTML files"
    temp_string = '<a name'
    file_len = len(src_not_tested_nlines)
    fileopen_error = [];
    ntotal_lines = 0
    ntotal_lines_not_tested = 0
    output_list = []
    nfiles_not_processed = 0
    sep = LOC+os.sep

    for file_ctr in range(0,file_len):
        inhtml_file = PETSC_DIR+os.sep+src_not_tested_path[file_ctr]+os.sep+src_not_tested_filename[file_ctr]+'.html'
        outhtml_file = LOC+os.sep+src_not_tested_path[file_ctr]+os.sep+src_not_tested_filename[file_ctr]+'.gcov.html'
        try:
            inhtml_fid = open(inhtml_file,"r")
        except IOError:
            # Error check for files not opened correctly or file names not parsed correctly in stage 1
            fileopen_error.append([src_not_tested_path[file_ctr],src_not_tested_filename[file_ctr]])
            nfiles_not_processed += 1
            continue

        temp_list = []
        temp_list.append(src_not_tested_filename[file_ctr])
        temp_list.append(string.split(outhtml_file,sep)[1]) # Relative path of hyperlink
        temp_list.append(src_not_tested_nlines[file_ctr])

        outhtml_fid = open(outhtml_file,"w")
        lines_not_tested = src_not_tested_lines[file_ctr]
        nlines_not_tested = src_not_tested_nlines[file_ctr]
        line_ctr = 0
        last_line_blank = 0
        for line in inhtml_fid:
            if(line.find(temp_string) != -1):
                nsrc_lines = int(line.split(':')[0].split('line')[1].split('"')[0].lstrip())
            if (line_ctr < nlines_not_tested):
                temp_line = 'line'+src_not_tested_lines[file_ctr][line_ctr]
                if (line.find(temp_line) != -1):
                    temp_outline = '<table><tr><td bgcolor="yellow">'+'<font size="4" color="red">!</font>'+line+'</td></table>'
                    line_ctr += 1
                else:
                    # Gcov information contains blank line numbers which C2HTML doesn't print, Need to handle this
                    # Marked line numbers 
                    if(line.find(temp_string) != -1):
                        line_num = int(line.split(':')[0].split('line')[1].split('"')[0].lstrip())

                        if (line_num > int(src_not_tested_lines[file_ctr][line_ctr])):
                            while (int(src_not_tested_lines[file_ctr][line_ctr]) < line_num):
                                line_ctr += 1
                                if(line_ctr == nlines_not_tested):
                                    last_line_blank = 1
                                    temp_outline = line
                                    break
                            if (last_line_blank == 0):        
                                temp_line = 'line'+src_not_tested_lines[file_ctr][line_ctr]
                                if(line.find(temp_line) != -1):
                                    temp_outline =  '<table><tr><td bgcolor="yellow">'+'<font size="4" color="red">!</font>'+line+'</td></table>'
                                    line_ctr += 1
                                else:
                                    temp_outline = line
                        else:
                            temp_outline = line
                    else:    
                        temp_outline = line
            else:
                temp_outline = line

            print >>outhtml_fid,temp_outline
            outhtml_fid.flush()

        inhtml_fid.close()
        outhtml_fid.close()

        ntotal_lines += nsrc_lines
        ntotal_lines_not_tested += src_not_tested_nlines[file_ctr]
        per_code_not_tested = float(src_not_tested_nlines[file_ctr])/float(nsrc_lines)*100.0

        temp_list.append(nsrc_lines)
        temp_list.append(per_code_not_tested)

        output_list.append(temp_list)

    shutil.rmtree(gcov_dir)
    # ------------------------------- End of Stage 3 ----------------------------------------

    # ------------------------------- Stage 4 ----------------------------------------------
    # Create Main HTML page containing statistics and marked HTML file links
    print "Creating main HTML page"
    # Create the main html file                                                                                                                                    
    # ----------------------------- index_gcov1.html has results sorted by file name ----------------------------------
    # ----------------------------- index_gcov2.html has results sorted by % code tested ------------------------------
    date_time = strftime("%x %X %Z")
    outfile_name1 = LOC+os.sep+'index_gcov1.html'
    outfile_name2 = LOC+os.sep+'index_gcov2.html'
    out_fid = open(outfile_name1,'w')                                            
    print >>out_fid, \
    """<html>                                                                                                                                                      
    <head>                                                                                                                                                         
      <title>PETSc:Code Testing Statistics</title>                                                                                                               
    </head>                                                                                                                                                        
    <body style="background-color: rgb(213, 234, 255);">"""                                                          
    print >>out_fid,"""<center>%s</center>"""%(date_time)
    print >>out_fid,"""<h2><center>Gcov statistics </center></h2>"""
    print >>out_fid,"""<center><font size = "4">Number of source code files = %s</font></center>""" %(nsrc_files)
    print >>out_fid,"""<center><font size = "4">Number of source code files not tested fully = %s</font></center>""" %(nsrc_files_not_tested)
    if float(nsrc_files) > 0: ratio = float(nsrc_files_not_tested)/float(nsrc_files)*100.0
    else: ratio = 0.0
    print >>out_fid,"""<center><font size = "4">Percentage of source code files not tested fully = %3.2f</font></center><br>""" %(ratio) 
    print >>out_fid,"""<center><font size = "4">Total number of source code lines = %s</font></center>""" %(ntotal_lines)
    print >>out_fid,"""<center><font size = "4">Total number of source code lines not tested = %s</font></center>""" %(ntotal_lines_not_tested)
    if float(ntotal_lines) > 0: ratio = float(ntotal_lines_not_tested)/float(ntotal_lines)*100.0
    else: ratio = 0.0
    print >>out_fid,"""<center><font size = "4">Percentage of source code lines not tested = %3.2f</font></center>""" %ratio
    print >>out_fid,"""<hr>    
    <a href = %s>See statistics sorted by percent code tested</a>""" % ('index_gcov2.html')
    print >>out_fid,"""<br><br>
    <h4><u><center>Statistics sorted by file name</center></u></h4>"""                                                        
    print >>out_fid,"""<table border="1" align = "center">                                                                                                                            
    <tr><th>Source Code</th><th>Lines in source code</th><th>Number of lines not tested</th><th>% Code not tested</th></tr>"""

    output_list.sort(key=operator.itemgetter(0),reverse=False)
    for file_ctr in range(0,nsrc_files_not_tested-nfiles_not_processed):
        print >>out_fid,"<tr><td><a href = %s>%s</a></td><td>%s</td><td>%s</td><td>%3.2f</td></tr>" % (output_list[file_ctr][1],output_list[file_ctr][0],output_list[file_ctr][3],output_list[file_ctr][2],output_list[file_ctr][4])

    print >>out_fid,"""</body>
    </html>"""
    out_fid.close()

    # ----------------------------- index_gcov2.html has results sorted by percentage code tested ----------------------------------                                        
    out_fid = open(outfile_name2,'w')                                                                                                    
    print >>out_fid, \
    """<html>                                                                                                     
    <head>                                                                                                                                                
      <title>PETSc:Code Testing Statistics</title>                                                             
    </head> 
    <body style="background-color: rgb(213, 234, 255);">"""                                                          
    print >>out_fid,"""<center>%s</center>"""%(date_time)
    print >>out_fid,"""<h2><center>Gcov statistics</center></h2>"""
    print >>out_fid,"""<center><font size = "4">Number of source code files = %s</font></center>""" %(nsrc_files)
    print >>out_fid,"""<center><font size = "4">Number of source code files not tested fully = %s</font></center>""" %(nsrc_files_not_tested)
    if float(nsrc_files) > 0: ratio = float(nsrc_files_not_tested)/float(nsrc_files)*100.0
    else: ratio = 0.0
    print >>out_fid,"""<center><font size = "4">Percentage of source code files not tested fully = %3.2f</font></center><br>""" %ratio
    print >>out_fid,"""<center><font size = "4">Total number of source code lines = %s</font></center>""" %(ntotal_lines)
    print >>out_fid,"""<center><font size = "4">Total number of source code lines not tested = %s</font></center>""" %(ntotal_lines_not_tested)
    if float(ntotal_lines) > 0: ratio = float(ntotal_lines_not_tested)/float(ntotal_lines)*100.0
    else: ratio = 0.0
    print >>out_fid,"""<center><font size = "4">Percentage of source code lines not tested = %3.2f</font></center>""" % ratio
    print >>out_fid,"""<hr>
    <a href = %s>See statistics sorted by file name</a>""" % ('index_gcov1.html') 
    print >>out_fid,"""<br><br>
    <h4><u><center>Statistics sorted by percent code tested</center></u></h4>"""
    print >>out_fid,"""<table border="1" align = "center">                                             
    <tr><th>Source Code</th><th>Lines in source code</th><th>Number of lines not tested</th><th>% Code not tested</th></tr>"""
    output_list.sort(key=operator.itemgetter(4),reverse=True)
    for file_ctr in range(0,nsrc_files_not_tested-nfiles_not_processed):
        print >>out_fid,"<tr><td><a href = %s>%s</a></td><td>%s</td><td>%s</td><td>%3.2f</td></tr>" % (output_list[file_ctr][1],output_list[file_ctr][0],output_list[file_ctr][3],output_list[file_ctr][2],output_list[file_ctr][4])

    print >>out_fid,"""</body>
    </html>"""
    out_fid.close()

    print "End of gcov script"
    print """See index_gcov1.html in %s""" % (LOC)
    return

################# Main Script ########################
import os,sys
PETSC_DIR = os.environ['PETSC_DIR']
USER = os.environ['USER']
gcov_dir = "/tmp/gcov-"+USER

if (sys.argv[1] == "-run_gcov"):
    print "Running gcov and creating tarball"
    run_gcov(PETSC_DIR,USER,gcov_dir)
    make_tarball(gcov_dir)
elif (sys.argv[1] == "-merge_gcov"):
    print "Creating main html page"
    # check to see if LOC is given
    if os.path.isdir(sys.argv[2]):
        print "Using %s to save the main HTML file pages" % (sys.argv[2])
        LOC = sys.argv[2]
        tarballs = sys.argv[3:]
    else:
        print "No Directory specified for saving main HTML file pages, using PETSc root directory"
        LOC = PETSC_DIR
        tarballs = sys.argv[2:]
    make_htmlpage(LOC,tarballs)

        
else:
    print "No or invalid option specified:"
    print "Usage: To run gcov and create tarball"
    print "         ./rungcov.py -run_gcov      "
    print "Usage: To create main html page"
    print "         ./rungcov.py -merge_gcov [LOC] tarballs"



