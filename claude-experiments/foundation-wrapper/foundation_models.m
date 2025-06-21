#import <Foundation/Foundation.h>
#include "foundation_models.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#import "FoundationModelsWrapper-Swift.h"

struct LanguageModelSession {
    void* wrapper;  // FoundationModelsWrapper* stored as void*
};

struct LanguageModelResponse {
    char* content;
};

struct StreamingResponse {
    void* swift_stream;
    bool has_more;
};

struct Tool {
    char* name;
    char* description;
    const char* (*call_function)(const char* arguments_json);
    void* swift_tool;  // CToolWrapper* stored as void*
};

LanguageModelSession* language_model_session_create(void) {
    @autoreleasepool {
        LanguageModelSession* session = malloc(sizeof(LanguageModelSession));
        if (!session) return NULL;
        
        FoundationModelsWrapper* wrapper = [[FoundationModelsWrapper alloc] init];
        session->wrapper = (__bridge_retained void*)wrapper;
        
        return session;
    }
}

LanguageModelSession* language_model_session_create_with_instructions(const char* instructions) {
    @autoreleasepool {
        if (!instructions) return language_model_session_create();
        
        LanguageModelSession* session = malloc(sizeof(LanguageModelSession));
        if (!session) return NULL;
        
        NSString* nsInstructions = [NSString stringWithUTF8String:instructions];
        FoundationModelsWrapper* wrapper = [[FoundationModelsWrapper alloc] initWithInstructions:nsInstructions];
        session->wrapper = (__bridge_retained void*)wrapper;
        
        return session;
    }
}

LanguageModelSession* language_model_session_create_with_use_case(ModelUseCase use_case) {
    @autoreleasepool {
        LanguageModelSession* session = malloc(sizeof(LanguageModelSession));
        if (!session) return NULL;
        
        NSInteger useCase = (use_case == MODEL_USE_CASE_CONTENT_TAGGING) ? 1 : 0;
        FoundationModelsWrapper* wrapper = [[FoundationModelsWrapper alloc] initWithUseCase:useCase];
        session->wrapper = (__bridge_retained void*)wrapper;
        
        return session;
    }
}

LanguageModelSession* language_model_session_create_with_tools(Tool** tools, size_t tool_count, const char* instructions) {
    // For now, generic tools are not fully implemented
    // If this is a weather tool, use the weather tool approach
    if (tool_count > 0 && tools && tools[0] && 
        strcmp(tools[0]->name, "getWeather") == 0) {
        return language_model_session_create_with_weather_tool(
            tools[0]->call_function, 
            instructions
        );
    }
    
    // Otherwise, fallback to instructions-only session
    return language_model_session_create_with_instructions(instructions);
}

LanguageModelSession* language_model_session_create_with_weather_tool(const char* (*weather_function)(const char* city), const char* instructions) {
    @autoreleasepool {
        LanguageModelSession* session = malloc(sizeof(LanguageModelSession));
        if (!session) return NULL;
        
        NSString* nsInstructions = instructions ? [NSString stringWithUTF8String:instructions] : nil;
        
        // Create the wrapper with the weather tool function
        FoundationModelsWrapper* wrapper = [[FoundationModelsWrapper alloc] 
            initWithWeatherToolFunction:^NSString*(NSString* city) {
                const char* result = weather_function([city UTF8String]);
                return result ? [NSString stringWithUTF8String:result] : @"No weather data available";
            }
            instructions:nsInstructions
        ];
        
        session->wrapper = (__bridge_retained void*)wrapper;
        
        return session;
    }
}

void language_model_session_destroy(LanguageModelSession* session) {
    if (session) {
        if (session->wrapper) {
            CFBridgingRelease(session->wrapper);
        }
        free(session);
    }
}

LanguageModelResponse* language_model_session_respond(LanguageModelSession* session, const char* prompt) {
    @autoreleasepool {
        if (!session || !prompt) return NULL;
        
        FoundationModelsWrapper* wrapper = (__bridge FoundationModelsWrapper*)(session->wrapper);
        NSString* nsPrompt = [NSString stringWithUTF8String:prompt];
        
        __block LanguageModelResponse* response = NULL;
        __block BOOL completed = NO;
        
        [wrapper respondTo:nsPrompt completion:^(NSString* content, NSString* error) {
            if (content && !error) {
                response = malloc(sizeof(LanguageModelResponse));
                if (response) {
                    const char* utf8Content = [content UTF8String];
                    size_t contentLength = strlen(utf8Content) + 1;
                    
                    response->content = malloc(contentLength);
                    if (response->content) {
                        strcpy(response->content, utf8Content);
                    } else {
                        free(response);
                        response = NULL;
                    }
                }
            }
            completed = YES;
        }];
        
        // Wait for completion (simple busy wait - not ideal for production)
        while (!completed) {
            [[NSRunLoop currentRunLoop] runMode:NSDefaultRunLoopMode beforeDate:[NSDate dateWithTimeIntervalSinceNow:0.01]];
        }
        
        return response;
    }
}

LanguageModelResponse* language_model_session_respond_generating(LanguageModelSession* session, const char* prompt, const char* struct_name) {
    // This would require more complex implementation with @Generable structs
    // For now, fallback to regular respond
    return language_model_session_respond(session, prompt);
}

StreamingResponse* language_model_session_stream_response(LanguageModelSession* session, const char* prompt) {
    // Streaming not implemented in this simple wrapper
    // Return NULL to indicate not supported
    return NULL;
}

StreamingResponse* language_model_session_stream_response_generating(LanguageModelSession* session, const char* prompt, const char* struct_name) {
    return language_model_session_stream_response(session, prompt);
}

const char* language_model_response_get_content(LanguageModelResponse* response) {
    return response ? response->content : NULL;
}

void language_model_response_destroy(LanguageModelResponse* response) {
    if (response) {
        free(response->content);
        free(response);
    }
}

bool language_model_session_is_responding(LanguageModelSession* session) {
    @autoreleasepool {
        if (!session) return false;
        
        FoundationModelsWrapper* wrapper = (__bridge FoundationModelsWrapper*)(session->wrapper);
        return wrapper.isResponding;
    }
}

const char* language_model_session_get_transcript(LanguageModelSession* session) {
    @autoreleasepool {
        if (!session) return NULL;
        
        FoundationModelsWrapper* wrapper = (__bridge FoundationModelsWrapper*)(session->wrapper);
        NSString* transcript = wrapper.transcript;
        
        // Note: This returns a temporary pointer that may become invalid
        // In a real implementation, you'd want to manage this memory properly
        return [transcript UTF8String];
    }
}

bool streaming_response_has_next(StreamingResponse* stream) {
    return stream ? stream->has_more : false;
}

const char* streaming_response_get_next(StreamingResponse* stream) {
    // Simplified implementation
    return NULL;
}

void streaming_response_destroy(StreamingResponse* stream) {
    if (stream) {
        free(stream);
    }
}

ModelAvailability system_language_model_get_availability(void) {
    @autoreleasepool {
        BOOL available = [FoundationModelsWrapper checkAvailability];
        return available ? MODEL_AVAILABILITY_AVAILABLE : MODEL_AVAILABILITY_UNAVAILABLE;
    }
}

const char* system_language_model_get_unavailability_reason(void) {
    @autoreleasepool {
        NSString* reason = [FoundationModelsWrapper getUnavailabilityReason];
        return reason ? [reason UTF8String] : NULL;
    }
}

char* language_model_feedback_attachment_encode_json(const LanguageModelFeedbackAttachment* feedback) {
    // Simplified JSON encoding
    // In a real implementation, you'd use a proper JSON library
    if (!feedback) return NULL;
    
    size_t buffer_size = 1024;
    char* json = malloc(buffer_size);
    if (!json) return NULL;
    
    snprintf(json, buffer_size, 
        "{\"sentiment\":%d,\"issues_count\":%zu}", 
        feedback->sentiment, 
        feedback->issues_count);
    
    return json;
}

Tool* tool_create(const char* name, const char* description, 
                  const char* (*call_function)(const char* arguments_json)) {
    if (!name || !description || !call_function) return NULL;
    
    Tool* tool = malloc(sizeof(Tool));
    if (!tool) return NULL;
    
    size_t name_len = strlen(name) + 1;
    size_t desc_len = strlen(description) + 1;
    
    tool->name = malloc(name_len);
    tool->description = malloc(desc_len);
    
    if (!tool->name || !tool->description) {
        free(tool->name);
        free(tool->description);
        free(tool);
        return NULL;
    }
    
    strcpy(tool->name, name);
    strcpy(tool->description, description);
    tool->call_function = call_function;
    tool->swift_tool = NULL;  // Simplified version
    
    return tool;
}

void tool_destroy(Tool* tool) {
    if (tool) {
        free(tool->name);
        free(tool->description);
        free(tool);
    }
}