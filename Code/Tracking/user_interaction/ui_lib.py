class raw_tracking_data_reader():
    """
    This class reads the output file generated 
    based on the tracking step. And structures
    what has been read in a standardised format.
    """
    def __init__(self, text_file_path):
        self.text_file_path = text_file_path
        self.read()

    def read(self):
        parsed_content = []
        with open(self.text_file_path, 'r') as text_file:
            text_content = text_file.read()
            splitted_content = text_content.split("#\n")
            for i in splitted_content:
                if len(i) < 1: continue
                block = []
                for j in i.split("\n"):
                    block.append(j.split(" "))
                parsed_content.append(block)
            text_file.close()
        self.data = parsed_content

class stage_1_modification_reader:
    """
    This class reads a modification file (stage 1).
    And structures what has been read in a standardised
    format usable for further modifications.
    """
    def __init__(self, text_file_path):
        self.text_file_path = text_file_path
        self.read()

    def read(self):
        parsed_content = []
        with open(self.text_file_path, 'r') as text_file:
            text_content = text_file.read()
            splitted_content = text_content.split("\n")
            for i in splitted_content:
                if len(i) < 1: continue
                parsed_content.append(i.split(" "))
            text_file.close()
        self.data = parsed_content

class data_writer:
    """
    writes the content of a block within 
    a destination file
    """
    def __init__(self, output_text_file_path):
        self.out_path = output_text_file_path
        # creates file if deosn't exist
        # erases file content if exists
        with open(self.out_path, "w") as temp:
            temp.close()
    
    def write(self, data):
        with open(self.out_path, "a") as output_file:
            for block in data:
                block_string = "#\n"
                for line in block:
                    block_string = f"{block_string}{line[0]} {line[1]} {line[2]} {line[3]} {line[4]}\n"
                if block_string == "#\n": block_string = f"{block_string}\n"
                output_file.write(block_string)
            output_file.close()

def edit_raw_output(RTD_reader, S1M_reader):
    """
    returns the modified structure to be written in
    a new file. By taking into account the modifications
    made from the edit_stageX.txt file
    """
    modified_data = []

    for block in RTD_reader.data:
        new_block = []
        for line in block:
            class_id = line[0]

            for i in S1M_reader.data:
                if class_id in i and i[0] != class_id:
                    class_id = i[0]

            new_line = [class_id] + line[1:]
            if len(new_line) != 5: continue
            new_block.append(new_line)
            
        modified_data.append(new_block)

    return modified_data

def draw_bbox_from_file(file_path, input_video_path, output_video_path):
    """
    draw the bounding boxes and class_id contained in <file_path>
    On the input video located at <input_video_path>
    The output video must be saved at <output_video_path>
    """

def perform_tracking(input_video_path, output_text_file_path):
    """
    Performs the tracking on <input_video_path>
    Saves the bounding boxes metadata whithin <output_text_file_path>
    """

if __name__ == "__main__":
    raw_path = "Code/Tracking/user_interaction/raw_output.txt"
    edit_path = "Code/Tracking/user_interaction/edit_stage1.txt"
    output_path = "Code/Tracking/user_interaction/output_stage_1.txt"

    raw_reader = raw_tracking_data_reader(raw_path)
    edit_reader = stage_1_modification_reader(edit_path)
    writer = data_writer(output_path)

    modified_data = edit_raw_output(raw_reader, edit_reader)  
    writer.write(modified_data)