#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "foundation_models.h"

const char* weather_tool_function(const char* input) {
    printf("🔧 WEATHER TOOL CALLED! Input: %s\n", input);
    return "The temperature in the requested city is 75°F with clear skies from the C tool function.";
}

int main() {
    printf("Foundation Models C Wrapper Test\n");
    printf("================================\n\n");

    // Test 1: Check model availability
    printf("1. Checking model availability...\n");
    ModelAvailability availability = system_language_model_get_availability();
    if (availability == MODEL_AVAILABILITY_AVAILABLE) {
        printf("✓ Model is available\n");
    } else {
        printf("✗ Model is unavailable: %s\n", system_language_model_get_unavailability_reason());
    }
    printf("\n");

    // Test 2: Create a basic session
    printf("2. Creating language model session...\n");
    LanguageModelSession* session = language_model_session_create();
    if (session) {
        printf("✓ Session created successfully\n");
    } else {
        printf("✗ Failed to create session\n");
        return 1;
    }
    printf("\n");

    // Test 3: Simple response
    printf("3. Testing simple response...\n");
    const char* prompt = "What's a good name for a trip to Japan? Respond only with a title";
    LanguageModelResponse* response = language_model_session_respond(session, prompt);
    
    if (response) {
        const char* content = language_model_response_get_content(response);
        printf("✓ Response received: %s\n", content ? content : "No content");
        language_model_response_destroy(response);
    } else {
        printf("✗ Failed to get response\n");
    }
    printf("\n");

    // Test 4: Session with instructions
    printf("4. Testing session with custom instructions...\n");
    LanguageModelSession* rhyme_session = language_model_session_create_with_instructions(
        "You are a helpful assistant who always responds in rhyme."
    );
    
    if (rhyme_session) {
        printf("✓ Rhyme session created\n");
        
        LanguageModelResponse* rhyme_response = language_model_session_respond(
            rhyme_session, "Write a short greeting"
        );
        
        if (rhyme_response) {
            const char* rhyme_content = language_model_response_get_content(rhyme_response);
            printf("✓ Rhyme response: %s\n", rhyme_content ? rhyme_content : "No content");
            language_model_response_destroy(rhyme_response);
        }
        
        language_model_session_destroy(rhyme_session);
    } else {
        printf("✗ Failed to create rhyme session\n");
    }
    printf("\n");

    // Test 5: Content tagging use case
    printf("5. Testing content tagging use case...\n");
    LanguageModelSession* tagging_session = language_model_session_create_with_use_case(MODEL_USE_CASE_CONTENT_TAGGING);
    
    if (tagging_session) {
        printf("✓ Content tagging session created\n");
        
        LanguageModelResponse* tag_response = language_model_session_respond(
            tagging_session, "I love hiking in the mountains and taking photos of wildlife"
        );
        
        if (tag_response) {
            const char* tag_content = language_model_response_get_content(tag_response);
            printf("✓ Tagging response: %s\n", tag_content ? tag_content : "No content");
            language_model_response_destroy(tag_response);
        }
        
        language_model_session_destroy(tagging_session);
    } else {
        printf("✗ Failed to create tagging session\n");
    }
    printf("\n");

    // Test 6: Multi-turn conversation
    printf("6. Testing multi-turn conversation...\n");
    LanguageModelResponse* first_response = language_model_session_respond(
        session, "Write a haiku about fishing"
    );
    
    if (first_response) {
        printf("✓ First haiku: %s\n", language_model_response_get_content(first_response));
        language_model_response_destroy(first_response);
        
        LanguageModelResponse* second_response = language_model_session_respond(
            session, "Do another one about golf"
        );
        
        if (second_response) {
            printf("✓ Second haiku: %s\n", language_model_response_get_content(second_response));
            language_model_response_destroy(second_response);
            
            // Check transcript
            const char* transcript = language_model_session_get_transcript(session);
            if (transcript) {
                printf("✓ Conversation transcript available (length: %zu)\n", strlen(transcript));
            }
        }
    }
    printf("\n");

    // Test 7: Weather tool functionality (using simple approach)
    printf("7. Testing weather tool functionality...\n");
    
    // Create session with weather tool using the simple approach
    LanguageModelSession* weather_session = language_model_session_create_with_weather_tool(
        weather_tool_function,
        "You help users with weather forecasts. Use the getWeather tool when users ask about weather."
    );
    
    if (weather_session) {
        printf("✓ Weather tool session created successfully\n");
        
        printf("Making request to use weather tool...\n");
        LanguageModelResponse* weather_response = language_model_session_respond(
            weather_session, "What is the weather in Paris? Use the getWeather tool to find out."
        );
        if (weather_response) {
            printf("✓ Weather response: %s\n", language_model_response_get_content(weather_response));
            language_model_response_destroy(weather_response);
        } else {
            printf("✗ Failed to get weather response\n");
        }
        
        language_model_session_destroy(weather_session);
    } else {
        printf("✗ Failed to create weather tool session\n");
    }
    printf("\n");

    // Test 7a: Tool creation (backward compatibility)
    printf("7a. Testing tool creation (backward compatibility)...\n");
    Tool* weather_tool = tool_create(
        "getWeather",
        "Retrieve the latest weather information for a city",
        weather_tool_function
    );
    
    if (weather_tool) {
        printf("✓ Weather tool created successfully\n");
        
        // Test tool with session (will use the weather tool approach)
        Tool* tools[] = {weather_tool};
        LanguageModelSession* tool_session = language_model_session_create_with_tools(
            tools, 1, "Help the user with weather forecasts."
        );

        if (tool_session) {
            printf("✓ Tool session created\n");
            
            printf("Making request to use weather tool via old API...\n");
            LanguageModelResponse* weather_response2 = language_model_session_respond(
                tool_session, "What is the weather in London? Use the getWeather tool to find out."
            );
            if (weather_response2) {
                printf("✓ Weather response via old API: %s\n", language_model_response_get_content(weather_response2));
                language_model_response_destroy(weather_response2);
            }
            
            language_model_session_destroy(tool_session);
        }
        
        tool_destroy(weather_tool);
    } else {
        printf("✗ Failed to create weather tool\n");
    }
    printf("\n");

    // Test 8: Feedback attachment
    printf("8. Testing feedback attachment...\n");
    LanguageModelFeedbackAttachment feedback = {0};
    feedback.sentiment = FEEDBACK_SENTIMENT_NEGATIVE;
    feedback.issues_count = 1;
    
    FeedbackIssue issue = {
        .category = FEEDBACK_ISSUE_INCORRECT,
        .explanation = "The response was not accurate"
    };
    feedback.issues = &issue;
    
    char* feedback_json = language_model_feedback_attachment_encode_json(&feedback);
    if (feedback_json) {
        printf("✓ Feedback JSON: %s\n", feedback_json);
        free(feedback_json);
    } else {
        printf("✗ Failed to encode feedback\n");
    }
    printf("\n");

    // Cleanup
    printf("9. Cleaning up...\n");
    language_model_session_destroy(session);
    printf("✓ Session destroyed\n");

    printf("\nAll tests completed!\n");
    return 0;
}