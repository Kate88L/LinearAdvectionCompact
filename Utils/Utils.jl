using JSON

# Save the current error to a JSON file
function save_last_error(error)
    data = Dict("error" => error)
    open("last_error.json", "w+") do file
        write(file, JSON.json(data))
    end
end

# Load the last error from the JSON file
function load_last_error()
    try
        open("last_error.json", "r") do file
            data = JSON.parse(String(read(file)))
            return data["error"]
        end
    catch
        return nothing  # If no file exists yet
    end
end