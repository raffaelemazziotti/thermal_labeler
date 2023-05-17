classdef LazyFileReader2< handle
    % LazyFileReader - A class for lazily reading lines from a file in MATLAB.
    %   This class provides methods to read lines from a file in a lazy
    %   manner, allowing efficient traversal of large files without loading
    %   the entire file into memory.

    properties
        fileID
        totalLines
        linePositions

        currentLineIndex

    end

    methods
        function obj = LazyFileReader2(filePath)
            obj.fileID = fopen(filePath, 'a+');

            obj.countLines();
            fseek(obj.fileID, 0, 'bof');
            obj.currentLineIndex = 1;
        end

        function countLines(obj)
            fseek(obj.fileID, 0, 'bof');
            obj.totalLines = 0;
            currentline = 1;
            obj.linePositions(currentline) = 0;
            while ~feof(obj.fileID)
                ln = fgetl(obj.fileID);
                if ln>0
                    obj.totalLines = obj.totalLines + 1;
                    currentline = currentline +1;
                    obj.linePositions(currentline)=ftell(obj.fileID);
                end
            end
        end

        function goToLine(obj, lineNumber)
            % goToLine - Moves to a specific line
            %   Usage: obj.goToLine(lineNumber)
            %     - lineNumber: Line number to move to

            obj.validateLineNumber(lineNumber);
            obj.currentLineIndex = lineNumber;
        end

        function line = readLine(obj, lineNumber)
            % readLine - Reads a specific line from the file
            %   Usage: line = obj.readLine(lineNumber)
            %     - lineNumber: Line number to read

            obj.validateLineNumber(lineNumber);
            fseek(obj.fileID, obj.linePositions(lineNumber), 'bof');
            line = fgetl(obj.fileID);
            obj.currentLineIndex = lineNumber;
        end

        function writeLine(obj,line)
            fseek(obj.fileID, 0, 'eof');
            fprintf(obj.fileID,'%s\n',line);
            obj.totalLines = obj.totalLines+1;
            obj.currentLineIndex=obj.totalLines;
        end

        function nextLine = getNextLine(obj)
            % getNextLine - Retrieves the next line
            %   Usage: nextLine = obj.getNextLine()

            obj.validateLineNumber(obj.currentLineIndex + 1);
            nextLine = obj.readLine(obj.currentLineIndex + 1);

        end

        function currentLine = getCurrentLine(obj)
            % getCurrentLine - Retrieves the current line
            %   Usage: currentLine = obj.getCurrentLine()

            currentLine = obj.readLine(obj.currentLineIndex);
        end

        function previousLine = getPreviousLine(obj)
            % getPreviousLine - Retrieves the previous line
            %   Usage: previousLine = obj.getPreviousLine()

            obj.validateLineNumber(obj.currentLineIndex - 1);
            previousLine = obj.readLine(obj.currentLineIndex-1);
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
        function validateLineNumber(obj, lineNumber)
            % validateLineNumber - Validates if the line number is within bounds

            if lineNumber < 1 || lineNumber > obj.totalLines
                error(['Invalid line number. The file contains ', num2str(obj.totalLines), ' lines.']);
            end
        end
    end


end

