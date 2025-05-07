from simsystem.objects import Job

def RM(jobs: list[Job]) -> Job | None:
    """
    Rate Monotonic Scheduling (RMS) algorithm.
    
    ARGUMENTS:
        jobs : list[Job]
            List of jobs to be scheduled.
    Returns:
        Job | None
            The job with the highest priority (lowest period).
    """
    if not jobs:
        return None
    return min(jobs, key=lambda job: job.priority)

def EDF(jobs: list[Job]) -> Job | None:
    """
    Earliest Deadline First (EDF) scheduling algorithm.
    
    ARGUMENTS:
        jobs : list[Job]
            List of jobs to be scheduled.
    Returns:
        list[Job]
            List of jobs sorted by their absolute deadline (ascending).
    """
    
    if not jobs:
        return None
    return min(jobs, key=lambda job: job.deadline)
