-- Postgres 15.4

-- Checking that an mark is being added for a course from the plan.
create function IsValidInsertToMarks() returns trigger as
$$
begin
if new.CourseId in (
    select
        CourseId
    from
        Students natural
        join Plan
    where
        StudentId = new.StudentId
) then return new;
end if;
raise exception 'Attempt to add a mark in a course not included to the plan';
end;
$$ language plpgsql;

-- Checking that there are no marks for the course being deleted from the plan.
create function IsValidDeleteFromPlan() returns trigger as
$$
begin
if old.CourseId not in (
	select
		CourseId
	from
	    Students S natural join Marks
	where
	    S.GroupId = old.GroupId
) then return old;
end if;
raise exception 'Attempt to remove a course that has marks from the plan';
end;
$$ language plpgsql;

create trigger NoExtraMarks before insert on Marks
for each row execute function IsValidInsertToMarks();

create trigger NoExtraMarks before delete on Plan
for each row execute function IsValidDeleteFromPlan();
