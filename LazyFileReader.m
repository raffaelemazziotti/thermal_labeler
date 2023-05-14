classdef LazyFileReader < handle
    % LazyFileReader - A class for lazily reading lines from a file in MATLAB.
    %   This class provides methods to read lines from a file in a lazy
    %   manner, allowing efficient traversal of large files without loading
    %   the entire file into memory.

    properties
        fileID
        totalLines
        currentLineIndex
    end

    methods
        function obj = LazyFileReader(filePath)
            % LazyFileReader - Constructor method
            %   Creates a new LazyFileReader object.
            %   Usage: obj = LazyFileReader(filePath)
            %     - filePath: File path as a string

            obj.fileID = fopen(filePath, 'r');
            obj.totalLines = obj.countLines();
            obj.currentLineIndex=1;
        end

        function line = readLine(obj, lineNumber)
            % readLine - Reads a specific line from the file
            %   Usage: line = obj.readLine(lineNumber)
            %     - lineNumber: Line number to read

            obj.validateLineNumber(lineNumber);

            fseek(obj.fileID, 0, 'bof');
            for i = 1:lineNumber
                line = fgetl(obj.fileID);
            end
        end

        function currentLine = getCurrentLine(obj)
            % getCurrentLine - Retrieves the current line
            %   Usage: currentLine = obj.getCurrentLine()

            currentLine = obj.readLine(obj.currentLineIndex);
        end

        function nextLine = getNextLine(obj)
            % getNextLine - Retrieves the next line
            %   Usage: nextLine = obj.getNextLine()

            obj.validateLineNumber(obj.currentLineIndex + 1);
            obj.currentLineIndex = obj.currentLineIndex + 1;
            nextLine = obj.readLine(obj.currentLineIndex);
            
        end

        function previousLine = getPreviousLine(obj)
            % getPreviousLine - Retrieves the previous line
            %   Usage: previousLine = obj.getPreviousLine()

            obj.validateLineNumber(obj.currentLineIndex - 1);
            obj.currentLineIndex = obj.currentLineIndex - 1;
            previousLine = obj.readLine(obj.currentLineIndex);
        end

        function goToLine(obj, lineNumber)
            % goToLine - Moves to a specific line
            %   Usage: obj.goToLine(lineNumber)
            %     - lineNumber: Line number to move to

            obj.validateLineNumber(lineNumber);
            obj.currentLineIndex = lineNumber;
        end

        function reset(obj)
            % reset - Resets the current line index to the first line
            obj.currentLineIndex = 1;
        end

        function close(obj)
            % close - Closes the file
            fclose(obj.fileID);
        end
    end

    methods (Access = private)
        function lineCount = countLines(obj)
            % countLines - Counts the total number of lines in the file

            fseek(obj.fileID, 0, 'eof');
            fileSize = ftell(obj.fileID);
            fseek(obj.fileID, 0, 'bof');
            data = fread(obj.fileID, fileSize, 'uint8');
            lineCount = sum(data == 10); % Counting newline characters
            obj.reset();
        end

        function validateLineNumber(obj, lineNumber)
            % validateLineNumber - Validates if the line number is within bounds

            if lineNumber < 1 || lineNumber > obj.totalLines
                error(['Invalid line number. The file contains ', num2str(obj.totalLines), ' lines.']);
            end
        end
    end
end
