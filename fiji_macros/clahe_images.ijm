setBatchMode("true");

#@ File (label = "Input directory", style = "directory") input
//#@ File (label = "Output directory", style = "directory") output

newDir = input + File.separator + "clahe" + File.separator;
if (File.exists(newDir))
   exit("Destination directory already exists; remove it and then run this macro again");
File.makeDirectory(newDir);


#@ String (label = "File suffix", value = ".tif") suffix

processFolder(input);

// function to scan folders/subfolders/files to find files with correct suffix
function processFolder(input) {

	list = getFileList(input);
	list = Array.sort(list);
	for (i = 0; i < list.length; i++) {
		if(File.isDirectory(input + File.separator + list[i])) //if it is a directory, then go into directory to process
			processFolder(input + File.separator + list[i]);
		if(endsWith(list[i], suffix))
			processFile(input, newDir, list[i]);
			
			
	}
}

function processFile(input, newDir, file) {

print("Processing: " +  file );

// Load Bio-Formats plugin
run("Bio-Formats Macro Extensions");

// Open input file using Bio-Formats
run("Bio-Formats Importer", "open=[" +  input + File.separator + file + "]");

run("Enhance Contrast", "saturated=0.35");

run("Enhance Local Contrast (CLAHE)", "blocksize=127 histogram=256 maximum=3 mask=*None* fast_(less_accurate)");

// Save current image as TIFF
saveAs("Tiff", newDir + File.separator + file );
            
// Get title of active image window
title = getTitle();


print("Saving to: " + newDir + File.separator + title);

close("*");

}


showStatus("Task complete");
beep();
showMessage("Task complete");