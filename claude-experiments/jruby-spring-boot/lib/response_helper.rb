require 'json'

module ResponseHelper
  def json_response(data, status = 200)
    @response.status = status
    @response.content_type = 'application/json'
    @response.body = data.to_json
  end

  def error_response(message, status = 400)
    json_response({ error: message }, status)
  end
end