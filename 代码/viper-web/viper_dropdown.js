function setOptions(chosen) {
    var selbox = document.modules.args;

    selbox.options.length = 0;
    if (chosen == " ") {
        selbox.options[selbox.options.length] = new Option('Select a Module First', ' ');

    }
    if (chosen == "apk") {
        selbox.options[selbox.options.length] = new Option('Help', 'help');
        selbox.options[selbox.options.length] = new Option('Info', 'info');
        selbox.options[selbox.options.length] = new Option('Permissions', 'perm');
        selbox.options[selbox.options.length] = new Option('File List', 'list');
        selbox.options[selbox.options.length] = new Option('All', 'all');
        selbox.options[selbox.options.length] = new Option('Decompile', 'dump');
    }

    if (chosen == "clamav") {
        selbox.options[selbox.options.length] = new Option('Scan', 'run');
    }

    if (chosen == "debup") {
        selbox.options[selbox.options.length] = new Option('Info', 'info');
        selbox.options[selbox.options.length] = new Option('Extract', 'extract');
    }

    if (chosen == "editdistance") {
        selbox.options[selbox.options.length] = new Option('Calculate', 'run');
    }

    if (chosen == "elf") {
        selbox.options[selbox.options.length] = new Option('Sections', 'sections');
        selbox.options[selbox.options.length] = new Option('Segments', 'segments');
        selbox.options[selbox.options.length] = new Option('Symobols', 'symbols');
        selbox.options[selbox.options.length] = new Option('Interpreter', 'interp');
        selbox.options[selbox.options.length] = new Option('Dynamic', 'dynamic');
    }

    if (chosen == "email") {
        selbox.options[selbox.options.length] = new Option('Envelope', 'envelope');
        selbox.options[selbox.options.length] = new Option('Attachments', 'attach');
        selbox.options[selbox.options.length] = new Option('Header', 'header');
        selbox.options[selbox.options.length] = new Option('Trace', 'trace');
        selbox.options[selbox.options.length] = new Option('TraceAll', 'traceall');
        selbox.options[selbox.options.length] = new Option('Spoof Check', 'spoof');
        selbox.options[selbox.options.length] = new Option('All', 'all');
        selbox.options[selbox.options.length] = new Option('Open Attachment', 'Open');
    }

    if (chosen == "exif") {
        selbox.options[selbox.options.length] = new Option('Extract Meta Data', 'run');
    }

    if (chosen == "fuzzy") {
        selbox.options[selbox.options.length] = new Option('Fuzzy Search', 'run');
    }

    if (chosen == "html") {
        selbox.options[selbox.options.length] = new Option('Scripts', 'scripts');
        selbox.options[selbox.options.length] = new Option('Links', 'links');
        selbox.options[selbox.options.length] = new Option('iFrames', 'iframe');
        selbox.options[selbox.options.length] = new Option('Embedded Objects', 'embed');
        selbox.options[selbox.options.length] = new Option('Images', 'images');
        selbox.options[selbox.options.length] = new Option('Dump Objects', 'dump');
    }

    if (chosen == "idx") {
        selbox.options[selbox.options.length] = new Option('Parse IDX', 'run');
    }

    if (chosen == "image") {
        selbox.options[selbox.options.length] = new Option('Submit To Ghiro', 'ghiro');
    }

    if (chosen == "jar") {
        selbox.options[selbox.options.length] = new Option('Parse Java Archive', 'run');
    }

    if (chosen == "office") {
        selbox.options[selbox.options.length] = new Option('MetaData', 'meta');
        selbox.options[selbox.options.length] = new Option('OLE Information', 'oleid');
        selbox.options[selbox.options.length] = new Option('Document Streams', 'streams');
        selbox.options[selbox.options.length] = new Option('Export Streams', 'export');
    }

    if (chosen == "pdf") {
        selbox.options[selbox.options.length] = new Option('PDF ID', 'id');
        selbox.options[selbox.options.length] = new Option('PDF Streams', 'streams');
    }

    if (chosen == "pe") {
        selbox.options[selbox.options.length] = new Option('Imports', 'imports');
        selbox.options[selbox.options.length] = new Option('Exports', 'exports');
        selbox.options[selbox.options.length] = new Option('Resources', 'res');
        selbox.options[selbox.options.length] = new Option('ImpHash', 'imp');
        selbox.options[selbox.options.length] = new Option('Compile Time', 'compile');
        selbox.options[selbox.options.length] = new Option('PEID', 'peid');
        selbox.options[selbox.options.length] = new Option('Digital Certificates', 'security');
        selbox.options[selbox.options.length] = new Option('Code Language', 'language');
        selbox.options[selbox.options.length] = new Option('Sections', 'sections');
        selbox.options[selbox.options.length] = new Option('PE Hash', 'pehash');
    }

    if (chosen == "rat") {
        selbox.options[selbox.options.length] = new Option('List Decoders', 'list');
        selbox.options[selbox.options.length] = new Option('Auto Detect', 'auto');
    }

    if (chosen == "reports") {
        selbox.options[selbox.options.length] = new Option('Find Reports On Malwr', 'malwr');
        selbox.options[selbox.options.length] = new Option('Find Reports On Anubis', 'anubis');
        selbox.options[selbox.options.length] = new Option('Find Reports on ThreatExchange', 'threat');
        selbox.options[selbox.options.length] = new Option('Find Reports On Joe Sandbox', 'joe');
        selbox.options[selbox.options.length] = new Option('Find Reports On metascan', 'meta');
    }

    if (chosen == "shellcode") {
        selbox.options[selbox.options.length] = new Option('Scan For Known Shellcode', 'run');
    }

    if (chosen == "strings") {
        selbox.options[selbox.options.length] = new Option('All Strings', 'all');
        selbox.options[selbox.options.length] = new Option('IP & Domain Strings', 'hosts');
    }

    if (chosen == "swf") {
        selbox.options[selbox.options.length] = new Option('Decompress SWF', 'decom');
    }

    if (chosen == "virustotal") {
        selbox.options[selbox.options.length] = new Option('Results', 'scan');
        selbox.options[selbox.options.length] = new Option('Submit', 'submit');
    }

    if (chosen == "xor") {
        selbox.options[selbox.options.length] = new Option('XOR', 'xor');
        selbox.options[selbox.options.length] = new Option('ROT', 'rot');
        selbox.options[selbox.options.length] = new Option('All', 'all');
        selbox.options[selbox.options.length] = new Option('Export', 'export');
    }

    if (chosen == "yara") {
        selbox.options[selbox.options.length] = new Option('Scan this file', 'scan');
        selbox.options[selbox.options.length] = new Option('Show all rules', 'rules');
    }

    if (chosen == "mycmd") {
        selbox.options[selbox.options.length] = new Option('CMD', 'run');
    }
    if (chosen == "cnn") {
        selbox.options[selbox.options.length] = new Option('?????????????????????', 'models_structure');
	selbox.options[selbox.options.length] = new Option('??????CNNMalware_Model1??????????????????', 'CNNMalware_Model1');
	selbox.options[selbox.options.length] = new Option('??????My_CNNMalware_Model1??????????????????', 'My_CNNMalware_Model1');
	selbox.options[selbox.options.length] = new Option('??????My_CNNMalware_lenet5??????????????????', 'My_CNNMalware_lenet5');
	selbox.options[selbox.options.length] = new Option('??????????????????????????????????????????', 'Custom_predict');
	selbox.options[selbox.options.length] = new Option('??????????????????????????????', 'upload');
    }
    if (chosen == "data_preprocess"){
	selbox.options[selbox.options.length] = new Option('??????????????????????????????','bin_to_img');
	selbox.options[selbox.options.length] = new Option('???????????????????????????','load_datas');
	selbox.options[selbox.options.length] = new Option('Extract_pe_Features','extract_pe_features');
	selbox.options[selbox.options.length] = new Option('Extract_Opcodes','extract_opcodes');
	selbox.options[selbox.options.length] = new Option('Count_Samples','count_samples');
	selbox.options[selbox.options.length] = new Option('Split_Opcodes','split_opcodes');
	selbox.options[selbox.options.length] = new Option('Latex_Format','latex_format');
    }
    if (chosen == "detect_malware"){
	selbox.options[selbox.options.length] = new Option('??????????????????????????????????????????','deep_feedforward');
	selbox.options[selbox.options.length] = new Option('Deep_Rnn','deep_rnn');
	selbox.options[selbox.options.length] = new Option('Shallow_Ml','shallow_ml');
	selbox.options[selbox.options.length] = new Option('Transfer_Conv_Ml','transfer_conv_ml');
    }

}
