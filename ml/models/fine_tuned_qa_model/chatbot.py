import json
import torch
from transformers import BertTokenizerFast, BertForQuestionAnswering
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from sklearn.model_selection import train_test_split

# Initialize tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Custom dataset
dataset = [



        {
      "question": "What should I do if I forget my password?",
      "context": "If you forget your password, click on the 'Forgot Password' link on the login page. Follow the instructions to reset it via your registered email.",
      "answer": "Click 'Forgot Password' and follow the reset instructions."
    },
    {
      "question": "How do I log into the platform?",
      "context": "To log into the platform, use your registered email and password. Make sure your credentials are correct to avoid login issues.",
      "answer": "Enter your registered email and password to log into the platform."
    },
    {
      "question": "How can I create a new project?",
      "context": "To start a new project, you need to be logged in and have the necessary permissions. Navigate to the dashboard and click on 'New Project' to begin.",
      "answer": "Go to the dashboard, click 'New Project', and provide the necessary details."
    },
    {
      "question": "Which departments can I collaborate with on the platform?",
      "context": "The platform supports collaboration among various departments. Departments such as Roads, Water Supply, Electricity, and Urban Development are part of the system.",
      "answer": "Departments such as Roads, Water Supply, Electricity, and Urban Development can collaborate on the platform."
    },
    {
      "question": "How do I add a team member to my project?",
      "context": "To add a team member, go to the project settings and navigate to the 'Manage Team' section. You will need their email to invite them to the project.",
      "answer": "In project settings, go to 'Manage Team', enter the member's email, and assign a role."
    },
    {
      "question": "How can I view the progress of a project?",
      "context": "The platform provides an overview of all ongoing projects. Check the 'Project Status' section for up-to-date information on the project's progress.",
      "answer": "Check 'Project Status' on the dashboard for progress details."
    },
    {
      "question": "What is the role of task dependencies?",
      "context": "Task dependencies ensure that tasks are completed in the correct order, preventing delays and ensuring the smooth execution of the project plan.",
      "answer": "Task dependencies ensure tasks are completed in the correct order."
    },
    {
      "question": "How do I upload files or documents to a project?",
      "context": "To share important documents with the team, go to the project details page, where you can find an option to upload files.",
      "answer": "Go to the project details page and use the 'Upload Files' option."
    },
    {
      "question": "How can I set task priorities?",
      "context": "Setting task priorities helps in managing the workflow efficiently. You can adjust task priorities from the task settings menu.",
      "answer": "Set task priorities to 'High', 'Medium', or 'Low' in task settings."
    },
    {
      "question": "How do I resolve resource conflicts?",
      "context": "Resource conflicts can occur when multiple tasks require the same resource. You can resolve this by redistributing resources or re-prioritizing tasks.",
      "answer": "Redistribute resources, re-prioritize tasks, or collaborate with departments to resolve conflicts."
    },
    {
      "question": "What are milestones in a project?",
      "context": "Milestones mark significant points in the project timeline. They are often used to track progress and assess the completion of key deliverables.",
      "answer": "Milestones are key events marking significant project progress."
    },
    {
      "question": "How do I get notified about project updates?",
      "context": "To stay informed, you can enable notifications in the platform settings. This will ensure you are updated about any changes or developments in your projects.",
      "answer": "Enable notifications in settings to receive updates."
    },
    {
      "question": "How do I track task deadlines?",
      "context": "Task deadlines can be tracked through the project timeline or task list. These features help you manage and keep track of the deadlines for each task.",
      "answer": "Track task deadlines in the project timeline or task list."
    },
    {
      "question": "How can I integrate this platform with other tools?",
      "context": "The platform offers integration options with tools like Slack and Trello. You can set up integrations from the 'Integrations' settings.",
      "answer": "Use 'Integrations' settings to connect tools like Slack and Trello."
    },
    {
      "question": "What happens if a task is delayed?",
      "context": "If a task is delayed, it will be flagged in the system. Team members will be notified, and adjustments can be made to the timeline to accommodate the delay.",
      "answer": "Delayed tasks are flagged, and notifications are sent to team members."
    },
    {
      "question": "How do I access project reports?",
      "context": "You can generate and download reports for your projects directly from the 'Reports' section in the platform.",
      "answer": "Go to 'Reports' and download project reports in the desired format."
    },
    {
      "question": "How are resources assigned to tasks?",
      "context": "Resources are assigned based on availability and task requirements. The platform helps match the right resources to the right tasks.",
      "answer": "Resources are assigned based on availability and task requirements."
    },
    {
      "question": "How do I request additional resources for a project?",
      "context": "In the 'Resource Management' section, you can submit requests for additional resources when you need them for your project.",
      "answer": "Use 'Resource Management' to submit additional resource requests."
    },
    {
      "question": "How do I resolve conflicts with other departments?",
      "context": "Interdepartmental conflicts can be resolved through communication, negotiation, and timeline adjustments. Collaboration is key to resolving these conflicts.",
      "answer": "Resolve conflicts through communication, negotiation, and timeline adjustments."
    },
    {
      "question": "What is the purpose of the discussion forum?",
      "context": "The discussion forum provides a space for team members to collaborate, share ideas, and resolve project-related issues.",
      "answer": "The forum is for team collaboration, discussions, and issue resolution."
    },
    {
      "question": "What is the role of the communication frequency?",
      "context": "Communication frequency ensures that team members are regularly updated about project progress and any changes in the project plan.",
      "answer": "Communication frequency ensures updates are shared regularly to keep the team informed about project progress."
    },
    {
      "question": "What departments are involved in the project?",
      "context": "The platform typically involves departments like Roads, Water Supply, Electricity, and Gas Pipelines in various projects.",
      "answer": "Departments such as Roads, Electricity, Water Supply, and Gas Pipelines are typically involved in a project."
    },
    {
      "question": "How is task priority determined?",
      "context": "Task priority is based on the urgency and importance of the task in relation to the project's overall goals.",
      "answer": "Task priority is determined based on the importance and urgency of the task in relation to project goals."
    },
    {
      "question": "What is resource optimization in project planning?",
      "context": "Resource optimization ensures that resources are used efficiently, minimizing waste and ensuring the project stays on track.",
      "answer": "Resource optimization ensures efficient use of resources to meet timelines and avoid overuse."
    },
    {
      "question": "What tools are used to monitor task progress?",
      "context": "Tools like Kanban boards, Gantt charts, and automated tracking systems are available to monitor the progress of tasks.",
      "answer": "Tools like Kanban boards, Gantt charts, and automated trackers are used to monitor task progress."
    },
    {
      "question": "How does communication delay impact a project?",
      "context": "Communication delays can lead to missed deadlines and confusion in task execution, causing delays in the overall project timeline.",
      "answer": "Communication delays can cause missed deadlines and unclear task execution."
    },
    {
      "question": "What is the significance of project milestones?",
      "context": "Project milestones represent significant achievements or stages in the project that help track its overall progress and success.",
      "answer": "Project milestones mark significant stages in a project and help track progress."
    },
    {
      "question": "How can resource conflicts be resolved?",
      "context": "Resource conflicts can be managed by reallocating resources, adjusting schedules, or negotiating with other departments to resolve the issue.",
      "answer": "Resource conflicts are resolved through prioritization, redistribution, and scheduling adjustments."
    },
    {
      "question": "What factors lead to project cost overruns?",
      "context": "Poor planning, unforeseen delays, and mismanagement of resources are common factors that lead to cost overruns in projects.",
      "answer": "Poor planning, delays, and resource mismanagement lead to cost overruns."
    },
    {
      "question": "How do I track resource utilization in a project?",
      "context": "Resource utilization can be tracked by monitoring the completion of tasks and comparing it to the resources allocated.",
      "answer": "Resource utilization is tracked using task completion rates and performance monitoring."
    },
    {
      "question": "How can I prioritize tasks in a project?",
      "context": "Prioritizing tasks helps ensure that the most critical tasks are completed first, helping to maintain the project's timeline and resources.",
      "answer": "Prioritize tasks based on deadlines, impact, and available resources."
    },
    {
      "question": "What is the role of the project manager?",
      "context": "The project manager is responsible for overseeing the project, coordinating team efforts, managing resources, and ensuring the project stays on track.",
      "answer": "The project manager oversees the project's execution, manages resources, and ensures deadlines are met."
    },
    {
      "question": "How do I track project performance?",
      "context": "Project performance can be tracked using various tools like performance reports, task completion rates, and milestone achievements.",
      "answer": "Track performance through reports, completion rates, and milestones."
    },
    {
      "question": "How do I submit a resource bidding request?",
      "context": "To submit a resource bidding request, go to the 'Resource Management' section and fill out the necessary details about the resources needed for the project.",
      "answer": "Go to 'Resource Biding' and fill out the form for resource bidding."
    },
    {
      "question": "How do I hire contractors for my project?",
      "context": "Hiring contractors involves reviewing available contractors in the 'Contractor Management' section and selecting the one that meets your project's needs.",
      "answer": "Review contractors in 'Contractor Management' and select the one that fits your project's requirements."
    },
    {
      "question": "How can I contact the Roads Department?",
      "context": "To contact the Roads Department, you can find their contact details under the 'Departments' section in the platform.",
      "answer": "Visit the 'Departments' section for contact details of the Roads Department."
    },
    {
      "question": "How do I check the availability of contractors for a project?",
      "context": "Contractors' availability can be checked by reviewing their schedule or contacting them directly through the platform.",
      "answer": "Check the contractor's availability in the 'Contractor Management' section or contact them directly."
    },
    {
      "question": "What should I do if I need to hire additional contractors?",
      "context": "If you need more contractors, you can request them through the 'Contractor Request' option in the platform.",
      "answer": "Use the 'Contractor Request' feature to hire additional contractors for your project."
    },
    {
      "question": "How can I resolve delays caused by contractors?",
      "context": "Delays caused by contractors can be managed by discussing the issue directly with them, adjusting timelines, or hiring additional contractors if necessary.",
      "answer": "Discuss delays with the contractor, adjust timelines, or hire more contractors if needed."
    },
    {
      "question": "Where can I find the contact information of the Water Supply Department?",
      "context": "You can find the contact information of all departments, including the Water Supply Department, in the 'Departments' section of the platform.",
      "answer": "Go to the 'Departments' section to find the contact details of the Water Supply Department."
    },
    {
      "question": "How do I add a contractor to the project?",
      "context": "To add a contractor, go to the project settings, select the 'Manage Contractors' section, and input the contractor's details.",
      "answer": "In project settings, navigate to 'Manage Contractors' and input the contractor's details."
    },
    {
      "question": "What is the process for releasing payments to contractors?",
      "context": "Payments to contractors can be processed through the 'Payments' section, where you can review invoices and approve payments.",
      "answer": "Use the 'Payments' section to review invoices and release payments to contractors."
    },
    {
      "question": "How do I manage resources that are allocated to contractors?",
      "context": "Resources allocated to contractors can be managed in the 'Resource Allocation' section, where you can adjust or redistribute resources as needed.",
      "answer": "Manage contractor resources through the 'Resource Allocation' section."
    },
    {
      "question": "How do I track project progress with external contractors?",
      "context": "Track the progress of external contractors by checking their assigned tasks, deadlines, and updates in the project dashboard.",
      "answer": "Track contractor progress by checking their tasks and updates in the project dashboard."
    },
    {
      "question": "Can I contact multiple departments through the platform?",
      "context": "Yes, the platform allows you to contact multiple departments by using the 'Department Communication' feature to send messages to specific teams.",
      "answer": "Use the 'Department Communication' feature to contact multiple departments at once."
    },
    {
      "question": "What if a contractor does not meet their deadlines?",
      "context": "If a contractor fails to meet deadlines, you can issue a warning or review the contract terms. The platform helps you monitor their performance.",
      "answer": "Issue a warning or review contract terms for contractors who miss deadlines."
    },
    {
      "question": "How do I find a contractor's previous work history?",
      "context": "Contractor work history is available in their profile under the 'Previous Projects' section, where you can review their past project performance.",
      "answer": "Review the 'Previous Projects' section in the contractor's profile for work history."
    },
    {
      "question": "How can I check the status of a contractor’s bid?",
      "context": "To check the status of a contractor's bid, go to the 'Contractor Bids' section, where you can see all submitted bids and their current statuses.",
      "answer": "Check the 'Contractor Bids' section to view the status of submitted bids."
    },
    {
      "question": "How do I communicate with contractors on the platform?",
      "context": "Contractors can be contacted directly through the 'Messaging' feature, where you can send private messages regarding project tasks.",
      "answer": "Use the 'Messaging' feature to communicate with contractors directly."
    },
    {
      "question": "What is the process for terminating a contract with a contractor?",
      "context": "Terminating a contract with a contractor can be done through the 'Contract Management' section, where you can review contract details and issue a termination.",
      "answer": "Terminate a contract through the 'Contract Management' section."
    },
    {
      "question": "How can I get the contact details of the Electricity Department?",
      "context": "Contact details for the Electricity Department are available in the 'Departments' section of the platform, under the specific department's profile.",
      "answer": "Visit the 'Departments' section for the contact details of the Electricity Department."
    },
    {
      "question": "How do I manage resource usage across different contractors?",
      "context": "You can manage resource usage for contractors by adjusting resource allocations through the 'Resource Allocation' section, ensuring resources are shared appropriately.",
      "answer": "Adjust resource allocations in the 'Resource Allocation' section to manage resource usage across contractors."
    },
    {
      "question": "How do I request a status update from contractors?",
      "context": "Request a status update by sending a message or using the 'Contractor Updates' feature to ask for progress reports on specific tasks.",
      "answer": "Send a message or use the 'Contractor Updates' feature to request status updates."
    },
    {
      "question": "How can I ensure contractors meet their project milestones?",
      "context": "To ensure contractors meet their milestones, set clear deadlines, track progress regularly, and provide feedback on their performance.",
      "answer": "Set clear deadlines, track progress, and provide feedback to ensure contractors meet milestones."
    },{
    "context": "The Smart City project for road repair and maintenance in Zone 5 started on March 1, 2024. The project is handled by Contractor ABC Pvt Ltd and is expected to be completed within 90 days. Resources include 20 workers, 3 road rollers, and 5 dump trucks.",
    "question": "Who is handling the Smart City project for road repair?",
    "answer": "Contractor ABC Pvt Ltd"
  },
  {
    "context": "The urban water supply project involves the construction of a 10 km pipeline to improve water distribution in the East sector. Currently, Phase 1 has been completed, and Phase 2 will begin next week.",
    "question": "What has been completed in the water supply project?",
    "answer": "Phase 1 has been completed"
  },
  {
    "context": "Project delay reports indicate that the construction of the flyover in Sector 8 has been delayed due to a shortage of steel beams. The revised deadline for the project is now December 31, 2024.",
    "question": "Why was the construction of the flyover delayed?",
    "answer": "Shortage of steel beams"
  },
  {
    "context": "The new park development project in Zone 12 includes landscaping, installing playground equipment, and setting up walking trails. The budget for this project is $500,000.",
    "question": "What is the budget for the park development project?",
    "answer": "Rs.500,000"
  },
  {
    "context": "The waste management initiative involves deploying 50 new garbage collection trucks across the city. Each truck is assigned a fixed route, and tracking systems have been installed for better efficiency.",
    "question": "How many garbage collection trucks are being deployed?",
    "answer": "50 trucks"
  },
  {
    "context": "The Housing Board's affordable housing project in Zone 7 aims to build 1,000 housing units for low-income families. Currently, 250 units have been completed, and 100 units are under construction.",
    "question": "How many housing units have been completed in the affordable housing project?",
    "answer": "250 units"
  },
  {
    "context": "The public transport improvement project includes adding 100 new electric buses to the city’s fleet. 40 buses have already been deployed on major routes, and the remaining buses will be operational by July 2024.",
    "question": "How many electric buses have been deployed so far?",
    "answer": "40 buses"
  }

    ]


    # Add more entries here

def preprocess_data(dataset, tokenizer, max_length=384):
    encodings = []
    start_positions = []
    end_positions = []

    for data in dataset:
        question = data["question"]
        context = data["context"]
        answer = data["answer"]

        # Find the answer's character-level start and end indices in the context
        start_idx = context.find(answer)
        if start_idx == -1:  # If answer is not found, print a warning and skip
            print(f"Warning: Answer '{answer}' not found in context: '{context}'")
            continue
        end_idx = start_idx + len(answer)

        # Tokenize with offset mapping
        encoding = tokenizer(
            question,
            context,
            truncation="only_second",  # Truncate only the context
            padding="max_length",
            max_length=max_length,
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        # Get token offsets and initialize start/end positions
        offsets = encoding["offset_mapping"][0]  # Get offsets for the first sequence pair
        start_pos, end_pos = None, None

        # Find token-level start and end positions based on character-level indices
        for idx, (start, end) in enumerate(offsets):
            if start <= start_idx < end and start_pos is None:
                start_pos = idx
            if start < end_idx <= end:
                end_pos = idx

        # If no valid start or end was found, assign default positions (CLS token index)
        if start_pos is None or end_pos is None:
            print(f"Problematic Example: \nQuestion: {question}\nContext: {context}\nAnswer: {answer}")
            start_pos = 0  # Default to CLS token
            end_pos = 0    # Default to CLS token

        # Append to encodings list
        encodings.append({
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0]
        })
        start_positions.append(start_pos)
        end_positions.append(end_pos)

    # Convert to tensors
    input_ids = torch.stack([e["input_ids"] for e in encodings])
    attention_masks = torch.stack([e["attention_mask"] for e in encodings])
    start_positions = torch.tensor(start_positions, dtype=torch.long)
    end_positions = torch.tensor(end_positions, dtype=torch.long)

    print("Finished processing the dataset!")
    return input_ids, attention_masks, start_positions, end_positions


# Tokenize the dataset
input_ids, attention_masks, start_positions, end_positions = preprocess_data(dataset, tokenizer)

# Check tokenized outputs
print("Input IDs shape:", input_ids.shape)
print("Attention Masks shape:", attention_masks.shape)
print("Start Positions:", start_positions)
print("End Positions:", end_positions)
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForQuestionAnswering, AdamW

# Prepare the TensorDataset and DataLoader
train_data = TensorDataset(input_ids, attention_masks, start_positions, end_positions)
train_loader = DataLoader(train_data, batch_size=2, shuffle=True)

# Initialize the model
chatbot_model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
chatbot_model.to(device)

# Define optimizer and training parameters
optimizer = AdamW(model.parameters(), lr=5e-5)
epochs = 15 # Adjust based on performance and dataset size

# Training loop
for epoch in range(epochs):
    chatbot_model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids_batch, attention_mask_batch, start_pos_batch, end_pos_batch = [b.to(device) for b in batch]

        optimizer.zero_grad()
        outputs = chatbot_model(
            input_ids=input_ids_batch,
            attention_mask=attention_mask_batch,
            start_positions=start_pos_batch,
            end_positions=end_pos_batch
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

# Save the trained model
chatbot_model.save_pretrained("fine_tuned_qa_model")
tokenizer.save_pretrained("fine_tuned_qa_model")
print("Model fine-tuning completed and saved!")
import re
import torch

def predict_answer(question, context, model, tokenizer, device):
    chatbot_model.eval()
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation="only_second",
        padding="max_length",
        max_length=384
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    start_idx = torch.argmax(start_logits)
    end_idx = torch.argmax(end_logits) + 1  # +1 to include the token

    # If the start index equals the end index, the model was not confident
    if start_idx == end_idx:
        return "No answer found...this query is out of basic knowledge. You can directly reach to concerned departments/authorities customer support management."

    # If valid start and end indices
    if start_idx <= end_idx:
        answer = tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx], skip_special_tokens=True)

        # Post-process to clean up the answer (remove spaces and commas)
        answer = answer.replace(",", "")  # Remove commas if present
        answer = re.sub(r'\s+', '', answer)  # Remove any extra spaces in the answer
        answer = answer.strip()  # Strip any extra leading/trailing whitespace

        return answer
    else:
        return "No answer found...this query is out of basic knowledge. You can directly reach to concerned departments/authorities customer support management."

# Test the model again
test_question = "How many residential units will be built in Zone 5?"
test_context = "The city’s urban development plan includes the construction of 500 new residential units in Zone 5. The project is expected to be completed in 2 years, with 150 units scheduled for completion in the first phase."

answer = predict_answer(test_question, test_context, chatbot_model, tokenizer, device)
print("Predicted Answer:", answer)
