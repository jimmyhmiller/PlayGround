#ifndef FOUNDATION_MODELS_H
#define FOUNDATION_MODELS_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LanguageModelSession LanguageModelSession;
typedef struct LanguageModelResponse LanguageModelResponse;
typedef struct StreamingResponse StreamingResponse;
typedef struct Tool Tool;

typedef enum {
    MODEL_AVAILABILITY_AVAILABLE,
    MODEL_AVAILABILITY_UNAVAILABLE
} ModelAvailability;

typedef enum {
    MODEL_USE_CASE_DEFAULT,
    MODEL_USE_CASE_CONTENT_TAGGING
} ModelUseCase;

typedef enum {
    FEEDBACK_SENTIMENT_POSITIVE,
    FEEDBACK_SENTIMENT_NEGATIVE,
    FEEDBACK_SENTIMENT_NEUTRAL
} FeedbackSentiment;

typedef enum {
    FEEDBACK_ISSUE_INCORRECT,
    FEEDBACK_ISSUE_INAPPROPRIATE,
    FEEDBACK_ISSUE_OTHER
} FeedbackIssueCategory;

typedef struct {
    FeedbackIssueCategory category;
    const char* explanation;
} FeedbackIssue;

typedef struct {
    const char** input;
    size_t input_count;
    const char** output;
    size_t output_count;
    FeedbackSentiment sentiment;
    FeedbackIssue* issues;
    size_t issues_count;
    const char*** desired_output_examples;
    size_t desired_output_examples_count;
} LanguageModelFeedbackAttachment;

LanguageModelSession* language_model_session_create(void);
LanguageModelSession* language_model_session_create_with_instructions(const char* instructions);
LanguageModelSession* language_model_session_create_with_use_case(ModelUseCase use_case);
LanguageModelSession* language_model_session_create_with_tools(Tool** tools, size_t tool_count, const char* instructions);
LanguageModelSession* language_model_session_create_with_weather_tool(const char* (*weather_function)(const char* city), const char* instructions);

void language_model_session_destroy(LanguageModelSession* session);

LanguageModelResponse* language_model_session_respond(LanguageModelSession* session, const char* prompt);
LanguageModelResponse* language_model_session_respond_generating(LanguageModelSession* session, const char* prompt, const char* struct_name);

StreamingResponse* language_model_session_stream_response(LanguageModelSession* session, const char* prompt);
StreamingResponse* language_model_session_stream_response_generating(LanguageModelSession* session, const char* prompt, const char* struct_name);

const char* language_model_response_get_content(LanguageModelResponse* response);
void language_model_response_destroy(LanguageModelResponse* response);

bool language_model_session_is_responding(LanguageModelSession* session);
const char* language_model_session_get_transcript(LanguageModelSession* session);

bool streaming_response_has_next(StreamingResponse* stream);
const char* streaming_response_get_next(StreamingResponse* stream);
void streaming_response_destroy(StreamingResponse* stream);

ModelAvailability system_language_model_get_availability(void);
const char* system_language_model_get_unavailability_reason(void);

char* language_model_feedback_attachment_encode_json(const LanguageModelFeedbackAttachment* feedback);

Tool* tool_create(const char* name, const char* description, 
                  const char* (*call_function)(const char* arguments_json));
void tool_destroy(Tool* tool);

#ifdef __cplusplus
}
#endif

#endif